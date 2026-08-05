# flake8: noqa: F401
"""
Topology 2 helpers for RoCEv2 TC7–TC11.

Topology 2 (2-Leaf + 1-Spine Clos, 8 tgen ports):
  Leaf0: 4 server-facing ports (tgen_port0..tgen_port3)
  Leaf1: 4 server-facing ports (tgen_port4..tgen_port7)
  Leaf0 ↔ Spine ↔ Leaf1  (uplinks; not tgen ports)

All 8 tgen ports are assumed to be connected to the DUTs.
The test fixture (`get_snappi_ports`, `fanout_graph_facts_multidut`,
`conn_graph_facts`) resolves which DUT each port belongs to.

Usage:
    from snappi_tests.rocev2.files.helper import *
    from snappi_tests.rocev2.files.helper_topo2 import (
        build_clos_ecmp_topology,
        build_spray_topology,
        ecmp_hash_verify,
        OOO_THRESHOLD,
    )

STC API gaps (CIPENH-1476, Topo2 track):
  - multi-DUT fixture path — verify against actual sonic-mgmt conftest
  - set_link_state (TC10) — field names TBC
  - out-of-order (OoO) counter field name — TBC
  - broadcast QP (TC11) — field name TBC
  All helpers mark unclear fields with # TODO-topo2.
"""
import logging

from tests.common.helpers.assertions import pytest_assert

logger = logging.getLogger(__name__)

# Out-of-order threshold: TC8 requires OoO packets < this fraction of total frames
OOO_THRESHOLD = 0.001   # 0.1%


def build_clos_ecmp_topology(port_ids, lossless_cfg_fn, lossless_spec, *,
                              src_leaf_port_count=4, dst_leaf_port_count=4):
    """
    Build a Clos ECMP topology dict compatible with configure_rocev2_topology().

    port_ids layout (caller order from get_snappi_ports):
      port_ids[0..src_leaf_port_count-1]  → Leaf0 server-facing (source ranks)
      port_ids[src_leaf_port_count..]      → Leaf1 server-facing (dest ranks)

    Generates one flow per (src_port, dst_port) pair so that all ECMP paths
    across Spine are exercised in TC7.

    lossless_cfg_fn(queue, message_size_kb) → dict (same contract as TC2–TC6).
    lossless_spec: list of (queue, message_size_kb) tuples — one per src-rank.

    Returns: (topology, src_port_ids, dst_port_ids)
    """
    pytest_assert(len(port_ids) >= src_leaf_port_count + dst_leaf_port_count,
                  f"Clos topology needs {src_leaf_port_count + dst_leaf_port_count} ports; "
                  f"got {len(port_ids)}")

    src_port_ids = port_ids[:src_leaf_port_count]
    dst_port_ids = port_ids[src_leaf_port_count:src_leaf_port_count + dst_leaf_port_count]

    topology = {}
    for i, src_pid in enumerate(src_port_ids):
        queue, message_size_kb = lossless_spec[i % len(lossless_spec)]
        cfg = lossless_cfg_fn(queue, message_size_kb)
        # One QP per destination port (multi-path ECMP exercise)
        qp_configs = []
        for j, dst_pid in enumerate(dst_port_ids):
            base_qp = dict(cfg.get("qp_configs", [{}])[0])  # template from first QP
            base_qp["qp_num"] = 100 + i * dst_leaf_port_count + j
            qp_configs.append(base_qp)
        entry = dict(cfg)
        entry["qp_configs"] = qp_configs
        entry["rx_port_ids"] = list(dst_port_ids)
        topology[src_pid] = entry

    return topology, src_port_ids, dst_port_ids


def build_spray_topology(port_ids, lossless_cfg_fn, lossless_spec, *,
                          src_leaf_port_count=4, dst_leaf_port_count=4):
    """
    Build Packet Spray topology (TC8): same Clos layout as build_clos_ecmp_topology
    but with `staggered_transmit` enabled to force per-packet spray across Spine links.

    Returns: (topology, src_port_ids, dst_port_ids)

    TODO-topo2: `staggered_transmit` field name — confirm against stc-x2rest
    multi-DUT support docs after Aug-8.
    """
    topology, src_port_ids, dst_port_ids = build_clos_ecmp_topology(
        port_ids, lossless_cfg_fn, lossless_spec,
        src_leaf_port_count=src_leaf_port_count,
        dst_leaf_port_count=dst_leaf_port_count)

    for src_pid in src_port_ids:
        port_cfg = topology[src_pid].get("rocev2_port_config", {})
        # TODO-topo2: field name may differ — "staggered_transmit", "packet_spray",
        # or a bool flag. Adjust when confirmed post Aug-8.
        port_cfg["staggered_transmit"] = True
        topology[src_pid]["rocev2_port_config"] = port_cfg

    return topology, src_port_ids, dst_port_ids


def ecmp_hash_verify(flow_df, dst_port_ids, *, imbalance_threshold=0.30):
    """
    Verify that traffic is spread across all dst_port_ids (ECMP load balancing).

    Checks:
      1. Every dst_port_id received at least some frames (all paths used).
      2. Rate imbalance across dst ports < imbalance_threshold.

    Args:
      flow_df: per-QP stats DataFrame (from collect_flow_queue_stats_portable).
      dst_port_ids: list of expected Rx port IDs.
      imbalance_threshold: max allowed (max-min)/max ratio.

    Returns nothing; raises AssertionError on failure.
    """
    if "port_rx" not in flow_df.columns or "frames_rx" not in flow_df.columns:
        logger.warning("ecmp_hash_verify: flow_df missing port_rx/frames_rx; skipping ECMP check")
        return

    rx_totals = flow_df.groupby("port_rx")["frames_rx"].sum()
    logger.info(f"ECMP Rx distribution:\n{rx_totals.to_string()}")

    for pid in dst_port_ids:
        rx_count = rx_totals.get(pid, 0)
        pytest_assert(rx_count > 0,
                      f"ECMP path {pid} received 0 frames — path not used (hash distribution broken)")

    counts = [rx_totals.get(pid, 0) for pid in dst_port_ids]
    max_c = max(counts)
    if max_c > 0:
        imbalance = (max_c - min(counts)) / max_c
        logger.info(f"ECMP Rx imbalance: {imbalance:.2%} (threshold: {imbalance_threshold:.0%})")
        pytest_assert(imbalance < imbalance_threshold,
                      f"ECMP imbalance {imbalance:.2%} >= {imbalance_threshold:.0%}; "
                      f"per-path counts: {dict(zip(dst_port_ids, counts))}")


def ooo_check(flow_df, *, threshold=OOO_THRESHOLD):
    """
    TC8 out-of-order check: out_of_order_rx / frames_rx < threshold.

    field 'out_of_order_rx' may not exist in early stc-x2rest versions.
    Logs a warning and skips the hard assertion if the field is absent.

    TODO-topo2: confirm field name against stc-x2rest post Aug-8.
    """
    ooo_col = "out_of_order_rx"   # TODO-topo2: field name TBC
    if ooo_col not in flow_df.columns:
        logger.warning(f"ooo_check: '{ooo_col}' not in flow_df — skipping OoO assertion. "
                       f"Confirm field name against stc-x2rest first version (CIPENH-1476).")
        return

    total_frames = flow_df["frames_rx"].sum()
    total_ooo = flow_df[ooo_col].sum()
    ooo_ratio = total_ooo / total_frames if total_frames > 0 else 0.0
    logger.info(f"OoO packets: {total_ooo}/{total_frames} = {ooo_ratio:.4%} "
                f"(threshold: {threshold:.2%})")
    pytest_assert(ooo_ratio < threshold,
                  f"Packet Spray OoO ratio {ooo_ratio:.4%} >= {threshold:.2%} "
                  f"(total_frames={total_frames}, ooo={total_ooo})")


def set_link_state_on_dut(duthost, interface, state):
    """
    Bring a DUT interface up or down for TC10 failover test.

    Args:
      duthost: a duthosts entry.
      interface: e.g. "Ethernet0"
      state: "up" or "down"

    TODO-topo2: `config interface` vs `ip link` — check DUT access method for
    multi-DUT Clos. Confirm sonic-mgmt multi-DUT fixture behavior for Spine.
    """
    if state == "down":
        cmd = f"sudo config interface shutdown {interface}"
    else:
        cmd = f"sudo config interface startup {interface}"
    logger.info(f"set_link_state: {duthost.hostname} {interface} → {state} via [{cmd}]")
    duthost.command(cmd)
