# flake8: noqa: F403, F401, F405
import pytest
import logging

from snappi_tests.rocev2.files.helper import *
from snappi_tests.rocev2.files.helper_tc2_portable import collect_flow_queue_stats_portable

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen', 'tgen')]

TRAFFIC_DURATION = 180
CNP_DSCP = 48
TX_RATE_PERCENT = 55
# DCQCN fairness threshold: per test plan, Tx rate imbalance between the two
# competing ranks must be < 30% of the higher rate.
FAIRNESS_THRESHOLD = 0.30


def test_rocev2_tc6_qp_fairness_dcqcn(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                ):
    """
    TC6 - QP Bandwidth Fairness under DCQCN (2:1 in-cast, ECN-capable).

    Same 2:1 in-cast topology as TC3 (dual-queue, ECN capable, DCQCN on),
    plus a bandwidth-fairness assertion: DCQCN must distribute throughput
    equitably between the two competing ranks.

    Fairness criterion (test plan):
      |TxRate_rank0 - TxRate_rank1| / max(TxRate_rank0, TxRate_rank1) < 30%

    All TC3 assertions also apply:
      - No packet loss, no NAK/sequence errors
      - ECN-CE and CNP must appear (DCQCN firing)
      - PFC informational only

    Key differentiator vs Ixia: vendor-neutral helper (no api._ixnetwork).
    Simplest remaining Topology 1 TC — pure TC3 + fairness check.

    References:
      Test plan: sonic-net/sonic-mgmt#21514
      TC1 reference: sonic-net/sonic-mgmt#23240
    """
    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 3, "Need minimum of 3 ports: 2 Tx ranks + 1 common Rx")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)
    queues = (3, 4)

    def conn(ack_nak_dscp):
        return {
            "choice": "reliable_connection",
            "reliable_connection": {
                "ack": {"ip_dscp": ack_nak_dscp, "ecn_value": "ect1"},
                "nak": {"ip_dscp": ack_nak_dscp, "ecn_value": "ect1"},
                "enable_retransmission_timeout": True,
                "retransmission_timeout_value": 5000,
                "retransmission_retry_count": 254,
            },
        }

    def lossless_cfg_fn(queue, message_size_kb):
        dscp = priority_to_dscp[queue]
        return {
            "mtu": 9100,
            "qp_configs": [{"message_size_unit": "kb", "message_size": message_size_kb,
                             "dscp": dscp, "ecn_capable": True}],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": True},
            "connection_type": conn(dscp),
            "rocev2_port_config": {"transmit_type": "target_line_rate", "target_line_rate": TX_RATE_PERCENT},
        }

    # dual_queue_uniform: same as TC3's dual_queue_uniform scenario
    lossless_spec = [(queues[0], 128), (queues[1], 128)]
    topology, rx_port_id, _, used_lossless = build_incast_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"TC6 Topology (2:1 in-cast ECN+DCQCN fairness, rx={rx_port_id}): {topology}")

    dsps = sorted(set(priority_to_dscp[q] for q in queues))
    qids = sorted(set(f"UC{q}" for q in queues))

    merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    pfc_df = pfc_counters(snappi_dut_port_map, direction="Tx")
    logger.info(f"DUT Tx PFC counters (informational):\n{tabulate(pfc_df, headers='keys', tablefmt='psql')}")

    # Fairness: per-rank Tx rate sampled during traffic.
    # build_rank_fairness returns a DataFrame with columns [port_tx, data_rx_rate_live]
    # (Rx rate at the common Rx port, grouped by originating Tx port).
    tx_port_ids = [p for p in port_ids if p != rx_port_id]
    rank_fair_df = build_rank_fairness(merged_df, dsps=dsps, by="port_tx")
    logger.info(f"TC6 per-rank Tx rates:\n{tabulate(rank_fair_df, headers='keys', tablefmt='psql')}")

    # Compute imbalance from the sampled rates
    rates = rank_fair_df["data_rx_rate_live"].values
    if len(rates) >= 2:
        max_rate = max(rates)
        imbalance = (max(rates) - min(rates)) / max_rate if max_rate > 0 else 0.0
        logger.info(f"TC6 rate imbalance: {imbalance:.2%} (threshold: {FAIRNESS_THRESHOLD:.0%})")
        assert imbalance < FAIRNESS_THRESHOLD, (
            f"DCQCN fairness violation: rank rate imbalance {imbalance:.2%} "
            f">= {FAIRNESS_THRESHOLD:.0%} — ranks: {dict(zip(tx_port_ids, rates))}"
        )

    checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "All messages complete, no loss", "message_fail=0 & frame_delta=0 required"),
        make_check("nak_tx != 0 or nak_rx != 0 or frame_sequence_error != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "nak_tx", "nak_rx", "frame_sequence_error"],
                "No NAK/sequence errors", "nak_tx=0 & nak_rx=0 & frame_sequence_error=0 required"),
        make_check(f"ip_dscp in {dsps} and ecn_ce_rx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "ecn_ce_rx"],
                f"ECN-CE observed on DSCPs {dsps} (DCQCN active)",
                f"ecn_ce_rx > 0 required — DCQCN must mark CE"),
        make_check(f"ip_dscp in {dsps} and cnp_tx == 0 and cnp_rx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "cnp_tx", "cnp_rx"],
                f"CNP observed on DSCPs {dsps}",
                f"cnp_tx > 0 or cnp_rx > 0 required — DUT must send CNP"),
    ]
    assert_queries(flow_df, checks)
    logger.info("*** TC6 PASSED ***")
