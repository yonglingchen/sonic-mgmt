# flake8: noqa: F403, F401, F405
import pytest
import logging

from snappi_tests.rocev2.files.helper import *
from snappi_tests.rocev2.files.helper_tc2_portable import collect_flow_queue_stats_portable
from snappi_tests.rocev2.files.helper_topo2 import (
    build_clos_ecmp_topology,
    set_link_state_on_dut,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen')]

TRAFFIC_DURATION = 60       # per segment (before flap, after flap)
FAILOVER_WAIT_SEC = 10      # wait for failover convergence after link down
RECOVERY_WAIT_SEC = 10      # wait for reconvergence after link up
CNP_DSCP = 48
TX_RATE_PERCENT = 30


def test_rocev2_tc10_failover_recovery(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                ):
    """
    TC10 - Failover & Recovery (Spine link flap during active traffic).

    Topology 2 (2-Leaf + 1-Spine Clos, 8 tgen ports).

    Test sequence:
      Phase 1 — Traffic runs over 2 Spine uplinks (ECMP, both active).
      Phase 2 — One Spine-to-Leaf0 link is shut down.
                 Traffic must reroute over remaining Spine paths.
                 RoCEv2 QPs must recover via retransmission.
      Phase 3 — Shutdown link is restored.
                 Traffic must re-converge back to full ECMP.

    Expected result at each phase:
      message_fail == 0 (all QPs complete) — retransmission timeout handles
      in-flight packets lost during link flap.
      No persistent sequence errors after reroute.

    The flap is applied to the Spine-to-Leaf0 uplink interface on the Spine DUT.
    In the 2-Leaf + 1-Spine Clos, duthosts[0]=Leaf0, duthosts[1]=Leaf1,
    duthosts[2]=Spine. If the fixture orders them differently, set the
    SPINE_DUT_INDEX and FLAP_INTERFACE below per testbed topology.

    TODO-topo2: multi-DUT fixture interface naming and duthosts ordering.
    Confirm duthosts list order against conftest for multidut-tgen topology.

    STC API note: set_link_state (OTG) vs sonic config interface shutdown —
    using DUT CLI (safer, no OTG version dependency). Confirm interface name
    on Spine DUT after physical testbed is assembled.

    Key differentiator vs Ixia: vendor-neutral helper (no api._ixnetwork).

    References:
      Test plan: sonic-net/sonic-mgmt#21514
      TC1 reference: sonic-net/sonic-mgmt#23240
    """
    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 8,
                   "TC10 Clos topology needs 8 tgen ports (4 Leaf0 + 4 Leaf1 server-facing)")
    pytest_require(len(duthosts) >= 3,
                   "TC10 needs 3 DUTs: Leaf0, Leaf1, Spine")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)
    queue = 3
    dscp = priority_to_dscp[queue]
    qids = [f"UC{queue}"]

    # TODO-topo2: adjust duthosts index and interface name to match actual testbed wiring.
    # Spine DUT is conventionally the 3rd DUT in multidut-tgen, with its Leaf0-facing
    # port named something like "Ethernet0" or derived from the fanout graph.
    SPINE_DUT_INDEX = 2
    SPINE_FLAP_IFACE = "Ethernet0"  # TODO-topo2: confirm from conn_graph_facts

    spine_dut = duthosts[SPINE_DUT_INDEX]

    def conn(ack_nak_dscp):
        return {
            "choice": "reliable_connection",
            "reliable_connection": {
                "ack": {"ip_dscp": ack_nak_dscp, "ecn_value": "non_ect"},
                "nak": {"ip_dscp": ack_nak_dscp, "ecn_value": "non_ect"},
                "enable_retransmission_timeout": True,
                "retransmission_timeout_value": 5000,
                "retransmission_retry_count": 254,
            },
        }

    def lossless_cfg_fn(q, message_size_kb):
        d = priority_to_dscp[q]
        return {
            "mtu": 9100,
            "qp_configs": [{"message_size_unit": "kb", "message_size": message_size_kb, "dscp": d}],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": False},
            "connection_type": conn(d),
            "rocev2_port_config": {"transmit_type": "target_line_rate",
                                   "target_line_rate": TX_RATE_PERCENT},
        }

    lossless_spec = [(queue, 128)] * 4
    topology, src_port_ids, dst_port_ids = build_clos_ecmp_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"TC10 Clos topology: src={src_port_ids} dst={dst_port_ids}")

    # Phase 1: baseline — both Spine uplinks active
    logger.info("TC10 Phase 1: baseline traffic (both Spine uplinks)")
    _merged_df, flow_df, _, _ = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=False)

    baseline_checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "Phase 1: zero loss before failover", "message_fail=0 & frame_delta=0"),
    ]
    assert_queries(flow_df, baseline_checks)

    # Phase 2: link flap — shut one Spine uplink, run traffic, verify reroute
    logger.info(f"TC10 Phase 2: shutting Spine interface {SPINE_FLAP_IFACE} on {spine_dut.hostname}")
    set_link_state_on_dut(spine_dut, SPINE_FLAP_IFACE, "down")
    wait_with_message(f"Waiting {FAILOVER_WAIT_SEC}s for failover convergence...", FAILOVER_WAIT_SEC)

    _merged_df2, flow_df2, _, _ = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=False)

    failover_checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "Phase 2: QPs recovered after Spine link flap (retransmission succeeded)",
                "message_fail=0 & frame_delta=0 — retransmission must absorb in-flight drops"),
        make_check("frame_sequence_error != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "frame_sequence_error"],
                "Phase 2: no persistent sequence errors after reroute",
                "frame_sequence_error=0 after failover converged"),
    ]
    assert_queries(flow_df2, failover_checks)

    # Phase 3: recovery — restore link, verify reconvergence
    logger.info(f"TC10 Phase 3: restoring Spine interface {SPINE_FLAP_IFACE}")
    set_link_state_on_dut(spine_dut, SPINE_FLAP_IFACE, "up")
    wait_with_message(f"Waiting {RECOVERY_WAIT_SEC}s for ECMP reconvergence...", RECOVERY_WAIT_SEC)

    _merged_df3, flow_df3, _, _ = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=False)

    recovery_checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "Phase 3: zero loss after Spine link recovery",
                "message_fail=0 & frame_delta=0 after reconvergence"),
    ]
    assert_queries(flow_df3, recovery_checks)
    logger.info("*** TC10 PASSED ***")
