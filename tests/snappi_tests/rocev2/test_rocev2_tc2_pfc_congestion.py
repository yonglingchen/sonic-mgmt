# flake8: noqa: F403, F401, F405
import pytest
import logging
import functools

from snappi_tests.rocev2.files.helper import *   # lib + re-exported fixtures
from snappi_tests.rocev2.files.helper_tc2_portable import (
    collect_flow_queue_stats_portable,
    override_rocev2_qp_verb,
    make_scale_qp_configs,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen', 'tgen')]

TRAFFIC_DURATION = 180      # seconds
CNP_DSCP = 48                # test plan: "DSCP for CNP value 48"
TX_RATE_PERCENT = 55         # test plan: "55% rate per Tx port"

# ---- traffic-pattern scenarios (test plan steps 3-8) -------------------
# queues:            (rank0 queue, rank1 queue) - both == same queue for the
#                    single_queue scenario (steps 3-4), since both ranks
#                    in-cast onto the SAME DSCP there.
# message_sizes_kb:  per-rank message size.
# verb_override:     rocev2_verb.choice to force on rank1's QP, or None to
#                    leave the default (send_with_immediate). Only
#                    dual_queue_mixed_size uses this ("WRITE only message"
#                    per the test plan step 7) - see override_rocev2_qp_verb()
#                    docstring for the schema-verification caveat.
SCENARIOS = {
    # steps 3-4: single queue, both ranks in-cast on DSCP for queue 3
    "single_queue": {
        "queues": (3, 3),
        "message_sizes_kb": (128, 128),
        "verb_override": None,
    },
    # steps 5-6: two queues, same message size on both ranks
    "dual_queue_uniform": {
        "queues": (3, 4),
        "message_sizes_kb": (128, 128),
        "verb_override": None,
    },
    # steps 7-8: two queues, rank1 uses a smaller WRITE-only message
    "dual_queue_mixed_size": {
        "queues": (3, 4),
        "message_sizes_kb": (128, 4),
        "verb_override": "write",
    },
}


@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_rocev2_tc2_pfc_congestion(
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
    TC2 - Congestion Control with PFC for AI Traffic (2:1 in-cast).

    Covers test plan steps 3-8 as three parametrized scenarios:
      single_queue          - steps 3-4: both ranks in-cast onto ONE queue
      dual_queue_uniform     - steps 5-6: two ranks, two queues, same msg size
      dual_queue_mixed_size  - steps 7-8: two ranks, two queues, rank1 uses a
                               smaller WRITE-only message

    In every scenario, each Tx rank sends stateful RoCEv2 traffic at
    TX_RATE_PERCENT of line rate into one common Rx rank, so the combined
    ingress on the Rx-side egress queue(s) exceeds line rate and PFC is
    expected. ECN-CE bit is left at its default ("00" / non_ect) and DCQCN is
    disabled - this test validates PFC congestion control in isolation;
    DCQCN's ECN/CNP response is TC3's job.

    Verifies (per RoCEv2_test_plan.md TC2 Expected Results):
      all messages complete with no loss, no NAK/sequence errors, latency
      within spec, ACK present on every driven queue (matching the DUT
      ACK-queue counter for its rank), PFC IS observed on every driven queue
      (the point of this test), no ECN-CE, and no CNP.

    Step 9 ("increase endpoints per port to 4K for testing T2") is a separate,
    heavier scale test - see test_rocev2_tc2_pfc_congestion_scale below.
    """
    spec = SCENARIOS[scenario_name]
    queues = spec["queues"]
    message_sizes_kb = spec["message_sizes_kb"]
    verb_override = spec["verb_override"]

    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 3, "Need minimum of 3 ports: 2 Tx ranks + 1 common Rx")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)

    # ---- traffic config (tune per test) -----------------------------------
    # Per test plan: ACK/NAK for each rank rides the SAME DSCP as that rank's
    # own data queue (unlike TC1, which used one dedicated shared ACK queue).
    def conn(ack_nak_dscp, ecn="non_ect"):
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
            "qp_configs": [{"message_size_unit": "kb", "message_size": message_size_kb, "dscp": dscp}],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": False},
            "connection_type": conn(dscp),
            "rocev2_port_config": {"transmit_type": "target_line_rate", "target_line_rate": TX_RATE_PERCENT},
        }

    # (queue, message_size_kb) per rank. single_queue gives ONE entry, which
    # build_incast_topology() cycles across BOTH Tx ranks (so both land on the
    # same queue/DSCP); the other two scenarios give one distinct entry per rank.
    if scenario_name == "single_queue":
        lossless_spec = [(queues[0], message_sizes_kb[0])]
    else:
        lossless_spec = [(queues[0], message_sizes_kb[0]), (queues[1], message_sizes_kb[1])]
    topology, rx_port_id, _, used_lossless = build_incast_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"[{scenario_name}] Test Topology (2:1 in-cast, rx={rx_port_id}): {topology}")

    # Tx ports in the order build_incast_topology assigned them (excludes rx).
    tx_port_ids = [p for p in port_ids if p != rx_port_id]

    post_configure_hook = None
    if verb_override is not None:
        # Only meaningful for the 2-distinct-queue scenarios; rank1 is the 2nd
        # Tx port. See override_rocev2_qp_verb() for the schema-verification
        # caveat on the "write" choice name.
        rank1_port_id = tx_port_ids[1]
        post_configure_hook = functools.partial(
            override_rocev2_qp_verb, tx_port_id=rank1_port_id, qp_index=0, verb_choice=verb_override)

    dsps = sorted(set(priority_to_dscp[q] for q in queues))
    qids = sorted(set(f"UC{q}" for q in queues))

    # ---- run + collect (portable - no IxNetwork-only calls; see helper_tc2_portable) ----
    # merged_df/sched_df (flow stats merged with DUT queue counters + scheduler
    # weights) aren't needed by this test's checks - kept unused here but
    # available if a later revision adds a DWRR/scheduler-based check.
    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True,
        post_configure_hook=post_configure_hook)

    # PFC counters are the authoritative signal (DUT `show pfc counters`), same
    # rationale as TC1: the snappi-side port stat is unreliable.
    pfc_df = pfc_counters(snappi_dut_port_map, direction="Tx")
    logger.info(f"[{scenario_name}] DUT Tx PFC counters:\n{tabulate(pfc_df, headers='keys', tablefmt='psql')}")
    driven_queues = sorted(set(queues))
    pfc_cols = [f"PFC{q}" for q in driven_queues if f"PFC{q}" in pfc_df.columns]
    # Failure condition (unlike TC1): PFC must be OBSERVED on EVERY queue we drove
    # into congestion - fail if any of them shows zero.
    pfc_fail_expr = " or ".join(f"{c} == 0" for c in pfc_cols) if pfc_cols else "1 == 1"

    # Per-rank ACK aggregate: sum ack_tx per port_tx, joined to that port's own
    # DUT ACK/NAK queue counter (ACK/NAK rides the SAME queue as data here).
    ack_rows = []
    for q in driven_queues:
        d = priority_to_dscp[q]
        sub = flow_df[flow_df["ip_dscp"] == d]
        agg = sub.groupby("port_tx", as_index=False)["ack_tx"].sum()
        agg["queue"] = f"UC{q}"
        agg[f"UC{q} totalpacket"] = agg["port_tx"].map(
            dut_queue_df.set_index("snappi_port").get(f"UC{q} totalpacket", {}))
        ack_rows.append(agg.rename(columns={f"UC{q} totalpacket": "dut_ack_ctr"}))
    ack_agg_df = pd.concat(ack_rows, ignore_index=True) if ack_rows else pd.DataFrame()

    # ---- checks (the part that defines what this test verifies) -----------
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
                f"avg_latency > {LATENCY_PROFILES['ai']['avg_latency_max']} is a failure"),
        # 4 - ACK present on every driven DSCP
        make_check(f"ip_dscp in {dsps} and ack_tx == 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "ack_tx", "ack_rx"],
                f"ACK on DSCPs {dsps}", f"ack_tx > 0 required for DSCPs {dsps}"),
        # 5 - no ECN-CE (ECN-CE bit left at 00 / non_ect for this test)
        make_check(f"ip_dscp in {dsps} and ecn_ce_rx != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "ecn_ce_rx"],
                f"No ECN-CE on DSCPs {dsps}", f"ecn_ce_rx == 0 required for DSCPs {dsps}"),
        # 6 - no CNP (DCQCN disabled)
        make_check(f"ip_dscp in {dsps} and (cnp_tx != 0 or cnp_rx != 0)",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "cnp_tx", "cnp_rx"],
                f"No CNP on DSCPs {dsps}", f"cnp_tx == 0 and cnp_rx == 0 required for DSCPs {dsps}"),
        # 7 - PFC IS observed on every driven queue (the point of this test -
        #     unlike TC1, absence of PFC here would be a FAIL, not a pass)
        make_check(pfc_fail_expr, pfc_cols or list(pfc_df.columns),
                f"PFC observed on queues {driven_queues} (congestion engaged)",
                f"DUT Tx PFC must be > 0 on every one of queues {driven_queues} under in-cast",
                override_df=pfc_df),
        # 8 - per-rank ACK matches that rank's own DUT ACK/NAK queue counter
        make_check("ack_tx != dut_ack_ctr",
                ["port_tx", "queue", "ack_tx", "dut_ack_ctr"],
                "Per-rank ACK matches DUT ACK/NAK queue counter",
                "sum(ack_tx) per (port_tx, queue) must equal that queue's DUT totalpacket counter",
                override_df=ack_agg_df),
    ]
    # Checks reference per-QP flow columns (ip_dscp, ack_tx, message_fail, ...)
    # plus two override_df checks (PFC from pfc_df, per-rank ACK from ack_agg_df).
    assert_queries(flow_df, checks)
    logger.info(f"*** TC2 [{scenario_name}] PASSED ***")


# ===========================================================================
# Step 9: scale variant - "Increase number of endpoint per port to 4K for
# testing T2". Kept as a separate, explicitly-marked test so it does not run
# by default alongside the 3 functional scenarios above - 4096 QP config
# objects per port is a materially heavier config-build/runtime than the rest
# of TC2, and the test plan itself scopes it to T2 role validation.
# ===========================================================================
NUM_ENDPOINTS_SCALE = 4096


# No custom pytest marker here - `scale` is not a marker registered in
# tests/pytest.ini, and adding an unregistered one risks an "unknown marker"
# warning/error depending on how strict-markers is configured for the run.
# Select/exclude this test by name instead, e.g. `-k "not scale"` /
# `-k test_rocev2_tc2_pfc_congestion_scale`.
def test_rocev2_tc2_pfc_congestion_scale(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                ):
    """
    TC2 step 9 - same 2:1 in-cast / 2-queue congestion scenario as
    dual_queue_uniform above, but with NUM_ENDPOINTS_SCALE QPs per rank
    instead of 1, per the test plan's "increase endpoints per port to 4K for
    testing T2".

    NOT YET VALIDATED at scale - see make_scale_qp_configs()'s docstring: config
    construction/set_config() time at full 4K has not been measured against a
    real backend. Start with a smaller NUM_ENDPOINTS_SCALE (e.g. 256) when
    first exercising this against real hardware, then raise it once the
    mechanism is confirmed to work.
    """
    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 3, "Need minimum of 3 ports: 2 Tx ranks + 1 common Rx")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)
    queues = (3, 4)

    def conn(ack_nak_dscp, ecn="non_ect"):
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
                NUM_ENDPOINTS_SCALE, dscp, message_size_kb=128, qp_num_base=100 + queue * 10000),
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": False},
            "connection_type": conn(dscp),
            "rocev2_port_config": {"transmit_type": "target_line_rate", "target_line_rate": TX_RATE_PERCENT},
        }

    lossless_spec = [(q,) for q in queues]
    topology, rx_port_id, _, used_lossless = build_incast_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"[scale] Test Topology (2:1 in-cast, {NUM_ENDPOINTS_SCALE} QPs/rank, rx={rx_port_id})")

    dsps = [priority_to_dscp[q] for q in queues]
    qids = [f"UC{q}" for q in queues]

    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    pfc_df = pfc_counters(snappi_dut_port_map, direction="Tx")
    pfc_cols = [f"PFC{q}" for q in queues if f"PFC{q}" in pfc_df.columns]
    pfc_fail_expr = " or ".join(f"{c} == 0" for c in pfc_cols) if pfc_cols else "1 == 1"

    # At 4K QPs/rank, per-QP checks are kept to aggregate-level (any-QP-failed)
    # rather than the full per-flow detail used in the functional scenarios,
    # to keep the failure report readable.
    checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "All messages complete, no loss (any QP)",
                "message_fail=0 & frame_delta=0 required for every QP"),
        make_check(pfc_fail_expr, pfc_cols or list(pfc_df.columns),
                f"PFC observed on queues {list(queues)} at scale",
                f"DUT Tx PFC must be > 0 on every one of queues {list(queues)}",
                override_df=pfc_df),
    ]
    assert_queries(flow_df, checks)
    logger.info(f"*** TC2 [scale, {NUM_ENDPOINTS_SCALE} QPs/rank] PASSED ***")
