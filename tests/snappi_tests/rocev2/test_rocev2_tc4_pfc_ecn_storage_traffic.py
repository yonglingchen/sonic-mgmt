# flake8: noqa: F403, F401, F405
import pytest
import logging

from snappi_tests.rocev2.files.helper import *
from snappi_tests.rocev2.files.helper_tc2_portable import (
    collect_flow_queue_stats_portable,
    make_scale_qp_configs,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen', 'tgen')]

TRAFFIC_DURATION = 180
CNP_DSCP = 48
# Storage traffic: constant-rate (not bursty), 1 MB message = ~256 packets at 4K MTU.
MESSAGE_SIZE_KB = 1024
# Constant bit rate target — storage workloads drive at a fixed line rate.
# TODO: confirm field name against stc-x2rest Aug-8 first version.
CONSTANT_BIT_RATE_GBPS = 100

# Two sub-runs: DCQCN off (PFC-dominant) then DCQCN on (ECN/CNP-dominant).
SCENARIOS = {
    # DCQCN disabled: PFC must fire; ECN-CE and CNP must NOT appear.
    "storage_no_dcqcn": {
        "queues": (3, 3),
        "dcqcn": False,
        "ecn_capable": False,
    },
    # DCQCN enabled: ECN-CE and CNP must appear; PFC should decrease.
    "storage_with_dcqcn": {
        "queues": (3, 3),
        "dcqcn": True,
        "ecn_capable": True,
    },
}


@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_rocev2_tc4_pfc_ecn_storage_traffic(
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
    TC4 - PFC + ECN/CNP Congestion Control for Storage Traffic.

    Storage workloads drive at constant rate with 1 MB messages (vs bursty AI
    traffic in TC2/TC3). Two parametrized sub-runs compare congestion behaviour
    with and without DCQCN:

      storage_no_dcqcn   - constant-rate 2:1 incast, DCQCN off.
                           PFC must fire; no ECN-CE or CNP expected.
      storage_with_dcqcn - same traffic, DCQCN on.
                           ECN-CE and CNP must appear; PFC decreases as
                           DCQCN throttles senders before PFC threshold.

    Key difference from TC2/TC3: constant_rate transmit mode; larger message.
    Key differentiator vs Ixia: vendor-neutral helper (no api._ixnetwork).

    References:
      Test plan: sonic-net/sonic-mgmt#21514
      TC1 reference: sonic-net/sonic-mgmt#23240
    """
    spec = SCENARIOS[scenario_name]
    queues = spec["queues"]
    dcqcn = spec["dcqcn"]
    ecn_capable = spec["ecn_capable"]

    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 3, "Need minimum of 3 ports: 2 Tx ranks + 1 common Rx")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)

    def conn(ack_nak_dscp):
        ecn = "ect1" if ecn_capable else "non_ect"
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
        qp_cfg = {"message_size_unit": "kb", "message_size": message_size_kb, "dscp": dscp}
        if ecn_capable:
            qp_cfg["ecn_capable"] = True
        return {
            "mtu": 9100,
            "qp_configs": [qp_cfg],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": dcqcn},
            "connection_type": conn(dscp),
            # constant_rate: fixed throughput regardless of congestion signals
            # (storage workload characteristic). Field names subject to
            # confirmation against stc-x2rest Aug-8 first version.
            "rocev2_port_config": {
                "transmit_type": "constant_rate",
                "constant_bit_rate": CONSTANT_BIT_RATE_GBPS,
            },
        }

    lossless_spec = [(queues[0], MESSAGE_SIZE_KB)]
    topology, rx_port_id, _, used_lossless = build_incast_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"[{scenario_name}] TC4 Topology (2:1 incast storage, dcqcn={dcqcn}, rx={rx_port_id})")

    driven_queues = sorted(set(queues))
    dsps = sorted(set(priority_to_dscp[q] for q in queues))
    qids = sorted(set(f"UC{q}" for q in queues))

    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    pfc_df = pfc_counters(snappi_dut_port_map, direction="Tx")
    pfc_cols = [f"PFC{q}" for q in driven_queues if f"PFC{q}" in pfc_df.columns]
    logger.info(f"[{scenario_name}] DUT Tx PFC counters:\n{tabulate(pfc_df, headers='keys', tablefmt='psql')}")

    checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "All messages complete, no loss", "message_fail=0 & frame_delta=0 required"),
        make_check("nak_tx != 0 or nak_rx != 0 or frame_sequence_error != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "nak_tx", "nak_rx", "frame_sequence_error"],
                "No NAK/sequence errors", "nak_tx=0 & nak_rx=0 & frame_sequence_error=0 required"),
        make_check(f"ip_dscp in {dsps} and ack_tx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "ack_tx", "ack_rx"],
                f"ACK on DSCPs {dsps}", f"ack_tx > 0 required for DSCPs {dsps}"),
    ]

    if not dcqcn:
        # DCQCN off: PFC must fire (same as TC2), no ECN-CE, no CNP
        pfc_fail_expr = " or ".join(f"{c} == 0" for c in pfc_cols) if pfc_cols else "1 == 1"
        checks += [
            make_check(pfc_fail_expr, pfc_cols or list(pfc_df.columns),
                    f"PFC fires on queues {driven_queues} (no DCQCN, PFC-dominant)",
                    "DUT Tx PFC must be > 0 without DCQCN", override_df=pfc_df),
            make_check(f"ip_dscp in {dsps} and ecn_ce_rx != 0",
                    ["flow_name", "port_tx", "port_rx", "ip_dscp", "ecn_ce_rx"],
                    f"No ECN-CE (DCQCN off, non-ECN traffic)",
                    f"ecn_ce_rx == 0 required when DCQCN disabled"),
            make_check(f"ip_dscp in {dsps} and (cnp_tx != 0 or cnp_rx != 0)",
                    ["flow_name", "port_tx", "port_rx", "ip_dscp", "cnp_tx", "cnp_rx"],
                    f"No CNP (DCQCN off)",
                    f"cnp_tx == 0 and cnp_rx == 0 required when DCQCN disabled"),
        ]
    else:
        # DCQCN on: ECN-CE and CNP must appear; PFC informational only
        checks += [
            make_check(f"ip_dscp in {dsps} and ecn_ce_rx == 0",
                    ["flow_name", "port_tx", "port_rx", "ip_dscp", "ecn_ce_rx"],
                    f"ECN-CE observed (DCQCN active on storage traffic)",
                    f"ecn_ce_rx > 0 required — DCQCN must mark CE"),
            make_check(f"ip_dscp in {dsps} and cnp_tx == 0 and cnp_rx == 0",
                    ["flow_name", "port_tx", "port_rx", "ip_dscp", "cnp_tx", "cnp_rx"],
                    f"CNP observed (DCQCN notification on storage traffic)",
                    f"cnp_tx > 0 or cnp_rx > 0 required — DUT must send CNP"),
        ]

    assert_queries(flow_df, checks)
    logger.info(f"*** TC4 [{scenario_name}] PASSED ***")
