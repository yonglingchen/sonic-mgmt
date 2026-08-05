# flake8: noqa: F403, F401, F405
import pytest
import logging

from snappi_tests.rocev2.files.helper import *
from snappi_tests.rocev2.files.helper_tc2_portable import (
    collect_flow_queue_stats_portable,
    make_scale_qp_configs,
)
from snappi_tests.rocev2.files.helper_topo2 import build_clos_ecmp_topology

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen')]

TRAFFIC_DURATION = 180
CNP_DSCP = 48
# 4:1 incast source rate — creates extreme congestion to trigger CP timeout scenario
TX_RATE_PERCENT = 95
# Retransmission parameters tuned for CP timeout test
RTX_TIMEOUT_MS = 1000       # tight timeout — fires quickly under congestion
RTX_RETRY_COUNT = 3         # max retries before giving up (expect 0 gives-up with mitigation)


def test_rocev2_tc11_cp_timeout_mitigation(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                ):
    """
    TC11 - Congestion Point Timeout Mitigation (4:1 incast + 1:4 broadcast).

    Topology 2 (2-Leaf + 1-Spine Clos, 8 tgen ports).

    Traffic pattern (simultaneous):
      4:1 incast:  all 4 Leaf0 ports → single Leaf1 port at 95% BW.
                   Creates extreme congestion → CP timeout scenario.
      1:4 broadcast: single Leaf0 port → all 4 Leaf1 ports at 30% BW each.
                   Exercises multicast/broadcast QP behavior under congestion.

    CP timeout mitigation:
      Without mitigation, extreme incast causes QP-level retransmission timeout
      and QP reset — traffic halts. With mitigation (adaptive timer backoff or
      DUT-side CP notification), the DUT must:
        - Allow QPs to complete under extreme congestion.
        - Zero message_fail on the 4:1 incast flows.
        - Zero message_fail on the 1:4 broadcast flows.

    STC API note:
      - broadcast QP (`broadcast_qp: True` in qp_configs) — field name TBC,
        confirmed HIGH risk per state.md Topo2 gaps. Test annotates with TODO.
      - 4:1 incast with 8 ports relies on rx_port_ids targeting one dest port
        from all 4 sources — confirm multi-target behavior in helper.py.

    Key differentiator vs Ixia: vendor-neutral helper (no api._ixnetwork).

    References:
      Test plan: sonic-net/sonic-mgmt#21514
      TC1 reference: sonic-net/sonic-mgmt#23240
    """
    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 8,
                   "TC11 Clos topology needs 8 tgen ports (4 Leaf0 + 4 Leaf1 server-facing)")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)
    queue = 3
    dscp = priority_to_dscp[queue]
    qids = [f"UC{queue}"]

    # Port assignment: Leaf0 = port_ids[0..3], Leaf1 = port_ids[4..7]
    leaf0_ports = port_ids[:4]
    leaf1_ports = port_ids[4:8]

    # Incast target: all Leaf0 ports aim at leaf1_ports[0]
    incast_target = leaf1_ports[0]
    # Broadcast source: leaf0_ports[0] aims at all Leaf1 ports
    bcast_source = leaf0_ports[0]

    def conn(ack_nak_dscp):
        return {
            "choice": "reliable_connection",
            "reliable_connection": {
                "ack": {"ip_dscp": ack_nak_dscp, "ecn_value": "non_ect"},
                "nak": {"ip_dscp": ack_nak_dscp, "ecn_value": "non_ect"},
                "enable_retransmission_timeout": True,
                "retransmission_timeout_value": RTX_TIMEOUT_MS,
                "retransmission_retry_count": RTX_RETRY_COUNT,
            },
        }

    def incast_cfg():
        return {
            "mtu": 9100,
            "qp_configs": [{"message_size_unit": "kb", "message_size": 128, "dscp": dscp}],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": True},    # DCQCN on — needed for extreme incast
            "connection_type": conn(dscp),
            "rocev2_port_config": {"transmit_type": "target_line_rate",
                                   "target_line_rate": TX_RATE_PERCENT},
            "rx_port_ids": [incast_target],              # 4:1 — all flows to one Rx port
        }

    def broadcast_cfg():
        # TODO-topo2: `broadcast_qp` field name TBC — HIGH risk per state.md Topo2 gaps.
        # If broadcast_qp is unsupported in first stc-x2rest version, substitute
        # with N unicast QPs (one per dest port) — same behavioral test.
        qp = {
            "message_size_unit": "kb",
            "message_size": 64,
            "dscp": dscp,
            # "broadcast_qp": True,   # TODO-topo2: uncomment when field name confirmed
        }
        return {
            "mtu": 9100,
            "qp_configs": [qp],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": False},
            "connection_type": conn(dscp),
            "rocev2_port_config": {"transmit_type": "target_line_rate",
                                   "target_line_rate": 30},
            "rx_port_ids": leaf1_ports,                  # 1:4 — one source to all Leaf1
        }

    # Build topology dict explicitly for TC11 (asymmetric layout)
    topology = {}
    for src_pid in leaf0_ports:
        topology[src_pid] = incast_cfg()
    # Overlay broadcast config on bcast_source (overrides its incast role)
    topology[bcast_source] = broadcast_cfg()

    logger.info(f"TC11 topology: incast_target={incast_target}, bcast_source={bcast_source}, "
                f"sources={leaf0_ports}")

    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    pfc_df = pfc_counters(snappi_dut_port_map, direction="Tx")
    logger.info(f"TC11 PFC counters (informational):\n{tabulate(pfc_df, headers='keys', tablefmt='psql')}")

    checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "All QPs complete under extreme incast + broadcast (CP timeout mitigation working)",
                "message_fail=0 & frame_delta=0 — CP timeout must not kill QPs"),
        make_check("nak_tx != 0 or nak_rx != 0 or frame_sequence_error != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "nak_tx", "nak_rx", "frame_sequence_error"],
                "No persistent NAK/sequence errors",
                "nak_tx=0 & nak_rx=0 & frame_sequence_error=0"),
    ]
    assert_queries(flow_df, checks)
    logger.info("*** TC11 PASSED ***")
