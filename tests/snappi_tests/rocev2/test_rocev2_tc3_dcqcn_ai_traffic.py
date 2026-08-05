# flake8: noqa: F403, F401, F405
import pytest
import logging

from snappi_tests.rocev2.files.helper import *   # lib + re-exported fixtures
from snappi_tests.rocev2.files.helper_tc2_portable import (
    collect_flow_queue_stats_portable,
    make_scale_qp_configs,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen', 'tgen')]

TRAFFIC_DURATION = 180      # seconds
CNP_DSCP = 48               # test plan: "DSCP for CNP value 48"
TX_RATE_PERCENT = 55        # test plan: "55% rate per Tx port" — same 2:1 in-cast as TC2

# ---- traffic-pattern scenarios (same topology as TC2, ECN/DCQCN enabled) ---
SCENARIOS = {
    # steps 3-4: single queue, both ranks in-cast on DSCP for queue 3
    "single_queue": {
        "queues": (3, 3),
        "message_sizes_kb": (128, 128),
    },
    # steps 5-6: two queues, same message size on both ranks
    "dual_queue_uniform": {
        "queues": (3, 4),
        "message_sizes_kb": (128, 128),
    },
    # steps 7-8: two queues, rank1 uses a smaller message
    "dual_queue_mixed_size": {
        "queues": (3, 4),
        "message_sizes_kb": (128, 4),
    },
}


@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_rocev2_tc3_dcqcn_ai_traffic(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                    scenario_name,
                                ):
    """
    TC3 - DCQCN Congestion Control for AI Traffic (2:1 in-cast, ECN-capable).

    Covers the same topology as TC2 (2:1 in-cast, Topology 1) but with ECN
    marking enabled and DCQCN active. The DUT must engage DCQCN — throttling
    the sender via ECN-CE marks and CNP packets — before PFC is needed.

    Three parametrized scenarios (same port layout as TC2):
      single_queue          - both ranks in-cast onto one queue (Q3)
      dual_queue_uniform    - two ranks, two queues, same message size
      dual_queue_mixed_size - two ranks, two queues, different message sizes

    Key difference from TC2:
      - ECN capable bit set on QP configs (ecn_capable=True)
      - DCQCN enabled (enable_dcqcn=True)
      - ECN-CE marks MUST appear on the congested flows (DCQCN firing)
      - CNP packets MUST appear (DUT sending back congestion notification)
      - PFC is expected to be absent or minimal (DCQCN throttles before PFC)
      - No packet loss and no NAK/sequence errors still required

    Key differentiator vs Ixia: uses helper_tc2_portable.py — no
    api._ixnetwork or StatViewAssistant calls. Works on any OTG backend
    (VIAVI TestCenter via Snappi API, IxNetwork, ixia-c).

    References:
      Test plan: sonic-net/sonic-mgmt#21514
      TC1 reference: sonic-net/sonic-mgmt#23240
    """
    spec = SCENARIOS[scenario_name]
    queues = spec["queues"]
    message_sizes_kb = spec["message_sizes_kb"]

    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 3, "Need minimum of 3 ports: 2 Tx ranks + 1 common Rx")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)

    # ---- traffic config: ECN-capable, DCQCN on --------------------------------
    # ACK/NAK also set ECN-capable so the DUT can mark them CE if needed,
    # matching the behaviour expected by DCQCN feedback loops.
    def conn(ack_nak_dscp, ecn="ect1"):
        return {
            "choice": "reliable_connection",
            "reliable_connection": {
                "ack": {"ip_dscp": ack_nak_dscp, "ecn_value": ecn},
                "nak": {"ip_dscp": ack_nak_dscp, "ecn_value": ecn},
                "enable_retransmission_timeout": True,
                "retransmission_timeout_value": 5000,
                "retransmission_retry_count": 254,
            },
        }

    def lossless_cfg_fn(queue, message_size_kb):
        dscp = priority_to_dscp[queue]
        return {
            "mtu": 9100,
            # ecn_capable=True: sets ECN-capable transport bit (01) in IP header,
            # enabling the DUT to mark CE on congested packets.
            "qp_configs": [{"message_size_unit": "kb", "message_size": message_size_kb,
                             "dscp": dscp, "ecn_capable": True}],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": True},
            "connection_type": conn(dscp),
            "rocev2_port_config": {"transmit_type": "target_line_rate", "target_line_rate": TX_RATE_PERCENT},
        }

    if scenario_name == "single_queue":
        lossless_spec = [(queues[0], message_sizes_kb[0])]
    else:
        lossless_spec = [(queues[0], message_sizes_kb[0]), (queues[1], message_sizes_kb[1])]

    topology, rx_port_id, _, used_lossless = build_incast_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"[{scenario_name}] Test Topology (2:1 in-cast ECN+DCQCN, rx={rx_port_id}): {topology}")

    driven_queues = sorted(set(queues))
    dsps = sorted(set(priority_to_dscp[q] for q in queues))
    qids = sorted(set(f"UC{q}" for q in queues))

    # ---- run + collect (portable, no IxNetwork-only calls) ---------------------
    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    # PFC: log for diagnostics but NOT a pass/fail requirement in TC3.
    # DCQCN is expected to throttle senders before PFC is needed; any PFC
    # observed is informational (DCQCN may not fully prevent it at high load).
    pfc_df = pfc_counters(snappi_dut_port_map, direction="Tx")
    logger.info(f"[{scenario_name}] DUT Tx PFC counters (informational — not a TC3 pass criterion):\n"
                f"{tabulate(pfc_df, headers='keys', tablefmt='psql')}")

    # ---- checks ---------------------------------------------------------------
    checks = [
        # 1 - all messages complete, no frame loss
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_tx", "message_complete_rx",
                 "message_fail", "frame_delta"],
                "All messages complete, no loss", "message_fail=0 & frame_delta=0 required"),
        # 2 - no NAK / sequence errors
        make_check("nak_tx != 0 or nak_rx != 0 or frame_sequence_error != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "nak_tx", "nak_rx", "frame_sequence_error"],
                "No NAK/sequence errors", "nak_tx=0 & nak_rx=0 & frame_sequence_error=0 required"),
        # 3 - latency within AI profile
        make_check(f"avg_latency > {LATENCY_PROFILES['ai']['avg_latency_max']}",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "avg_latency", "max_latency"],
                "Latency within spec",
                f"avg_latency <= {LATENCY_PROFILES['ai']['avg_latency_max']} required"),
        # 4 - ACK present on every driven DSCP
        make_check(f"ip_dscp in {dsps} and ack_tx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "ack_tx", "ack_rx"],
                f"ACK on DSCPs {dsps}", f"ack_tx > 0 required for DSCPs {dsps}"),
        # 5 - ECN-CE MUST appear (DCQCN is firing — DUT marked congested packets)
        make_check(f"ip_dscp in {dsps} and ecn_ce_rx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "ecn_ce_rx"],
                f"ECN-CE observed on DSCPs {dsps} (DCQCN active)",
                f"ecn_ce_rx > 0 required for DSCPs {dsps} — DCQCN must mark CE"),
        # 6 - CNP MUST appear (DUT sent congestion notification packets back)
        make_check(f"ip_dscp in {dsps} and cnp_tx == 0 and cnp_rx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "cnp_tx", "cnp_rx"],
                f"CNP observed on DSCPs {dsps} (DCQCN notification active)",
                f"cnp_tx > 0 or cnp_rx > 0 required for DSCPs {dsps} — DUT must send CNP"),
    ]
    assert_queries(flow_df, checks)
    logger.info(f"*** TC3 [{scenario_name}] PASSED ***")


# ===========================================================================
# Scale variant — 4K QPs per rank (test plan step 9 equivalent for TC3).
# ===========================================================================
NUM_ENDPOINTS_SCALE = 4096


def test_rocev2_tc3_dcqcn_ai_traffic_scale(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                ):
    """
    TC3 scale variant — same 2:1 in-cast / dual-queue DCQCN scenario with
    NUM_ENDPOINTS_SCALE QPs per rank. Validates that DCQCN operates correctly
    under 4K QP load (ECN-CE and CNP must still appear; no loss allowed).

    Run with: pytest ... -k test_rocev2_tc3_dcqcn_ai_traffic_scale
    Start with a smaller NUM_ENDPOINTS_SCALE (e.g. 256) on first hardware run.
    """
    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 3, "Need minimum of 3 ports: 2 Tx ranks + 1 common Rx")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)
    queues = (3, 4)

    def conn(ack_nak_dscp, ecn="ect1"):
        return {
            "choice": "reliable_connection",
            "reliable_connection": {
                "ack": {"ip_dscp": ack_nak_dscp, "ecn_value": ecn},
                "nak": {"ip_dscp": ack_nak_dscp, "ecn_value": ecn},
                "enable_retransmission_timeout": True,
                "retransmission_timeout_value": 5000,
                "retransmission_retry_count": 254,
            },
        }

    def lossless_cfg_fn(queue):
        dscp = priority_to_dscp[queue]
        return {
            "mtu": 9100,
            "qp_configs": make_scale_qp_configs(
                NUM_ENDPOINTS_SCALE, dscp, message_size_kb=128,
                qp_num_base=100 + queue * 10000,
                extra_qp_fields={"ecn_capable": True}),
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": True},
            "connection_type": conn(dscp),
            "rocev2_port_config": {"transmit_type": "target_line_rate", "target_line_rate": TX_RATE_PERCENT},
        }

    lossless_spec = [(q,) for q in queues]
    topology, rx_port_id, _, used_lossless = build_incast_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"[scale] TC3 Topology (2:1 in-cast ECN+DCQCN, {NUM_ENDPOINTS_SCALE} QPs/rank, rx={rx_port_id})")

    dsps = [priority_to_dscp[q] for q in queues]
    qids = [f"UC{q}" for q in queues]

    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "All messages complete, no loss (any QP)",
                "message_fail=0 & frame_delta=0 required for every QP"),
        make_check(f"ip_dscp in {dsps} and ecn_ce_rx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "ecn_ce_rx"],
                f"ECN-CE observed at scale on DSCPs {dsps}",
                f"ecn_ce_rx > 0 required — DCQCN must mark CE even at {NUM_ENDPOINTS_SCALE} QPs/rank"),
        make_check(f"ip_dscp in {dsps} and cnp_tx == 0 and cnp_rx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "cnp_tx", "cnp_rx"],
                f"CNP observed at scale on DSCPs {dsps}",
                f"cnp_tx > 0 or cnp_rx > 0 required at {NUM_ENDPOINTS_SCALE} QPs/rank"),
    ]
    assert_queries(flow_df, checks)
    logger.info(f"*** TC3 [scale, {NUM_ENDPOINTS_SCALE} QPs/rank] PASSED ***")
