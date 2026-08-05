# flake8: noqa: F403, F401, F405
import pytest
import logging

from snappi_tests.rocev2.files.helper import *
from snappi_tests.rocev2.files.helper_tc2_portable import collect_flow_queue_stats_portable

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen', 'tgen')]

TRAFFIC_DURATION = 180
CNP_DSCP = 48
TX_RATE_PERCENT = 100       # 100% BW per test plan
# Tester generates PFC pause at 90% BW toward DUT Rx ports.
# TODO: confirm pfc_gen_rate field name against stc-x2rest Aug-8 first version.
PFC_GEN_RATE_PERCENT = 90


def test_rocev2_tc5_pfc_propagation(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                ):
    """
    TC5 - PFC Pause Frame Propagation.

    Verifies that the DUT correctly propagates PFC pause frames upstream when
    its downstream receiver signals congestion.

    Traffic pattern (1:1, not 2:1 incast):
      rank0 (tgen_port1) → DUT → rank2 (tgen_port3) at 100% BW
      rank1 (tgen_port2) → DUT → rank3 (tgen_port4) at 100% BW

    Tester action:
      rank2 and rank3 generate PFC pause frames at ~90% BW toward the DUT's
      Rx-side ports. The DUT must absorb the pause and propagate it upstream.

    Expected result:
      PFC pause frames appear on the Tx-side (rank0, rank1) — proving the DUT
      forwarded the PFC upstream rather than absorbing it silently.

    STC API note:
      pfc_gen_enable + pfc_gen_rate in Rocev2EndpointConfig are marked medium-risk;
      confirm field names against stc-x2rest Aug-8 first version (CIPENH-1476).
      If pfc_gen_enable is absent in first version, a separate stateless PFC flow
      (OTG Flow with custom Ethernet/PFC payload) can substitute.

    Key differentiator vs Ixia: vendor-neutral helper (no api._ixnetwork).

    References:
      Test plan: sonic-net/sonic-mgmt#21514
      TC1 reference: sonic-net/sonic-mgmt#23240
    """
    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 4,
                   "TC5 needs 4 ports: rank0/rank1 (Tx) + rank2/rank3 (Rx + PFC gen)")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)

    queue = 3
    dscp = priority_to_dscp[queue]
    qids = [f"UC{queue}"]

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

    # 1:1 pairwise: port_ids[0]→port_ids[2], port_ids[1]→port_ids[3]
    # build_pairwise_topology creates one QP pair per adjacent port pair.
    common_cfg = {
        "mtu": 9100,
        "qp_configs": [{"message_size_unit": "kb", "message_size": 128, "dscp": dscp}],
        "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
        "dcqcn_settings": {"enable_dcqcn": False},
        "connection_type": conn(dscp),
        "rocev2_port_config": {"transmit_type": "target_line_rate", "target_line_rate": TX_RATE_PERCENT},
        # PFC generation on Rx-side ports (rank2, rank3) toward DUT.
        # These fields are in Rocev2EndpointConfig; field names TBC post Aug-8.
        "endpoint_config": {
            "pfc_gen_enable": True,
            "pfc_gen_rate": PFC_GEN_RATE_PERCENT,
            "pfc_gen_queue": queue,
        },
    }

    topology = build_pairwise_topology(port_ids, common_cfg)
    logger.info(f"TC5 Topology (1:1 pairwise PFC propagation): {topology}")

    tx_port_ids = port_ids[:2]   # rank0, rank1
    rx_port_ids = port_ids[2:4]  # rank2, rank3 (PFC generators)

    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    # PFC propagation check: DUT Tx-side ports (rank0, rank1) must show PFC.
    # The PFC originates from tester Rx ports (rank2, rank3) and should be
    # forwarded upstream by the DUT.
    pfc_tx_df = pfc_counters(snappi_dut_port_map, direction="Tx")
    pfc_rx_df = pfc_counters(snappi_dut_port_map, direction="Rx")
    logger.info(f"DUT Tx PFC (must be > 0 — propagated upstream):\n"
                f"{tabulate(pfc_tx_df, headers='keys', tablefmt='psql')}")
    logger.info(f"DUT Rx PFC (reference — generated by tester):\n"
                f"{tabulate(pfc_rx_df, headers='keys', tablefmt='psql')}")

    pfc_col = f"PFC{queue}"
    pfc_fail_expr = f"{pfc_col} == 0" if pfc_col in pfc_tx_df.columns else "1 == 1"

    checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "All messages complete, no loss", "message_fail=0 & frame_delta=0 required"),
        make_check("nak_tx != 0 or nak_rx != 0 or frame_sequence_error != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "nak_tx", "nak_rx", "frame_sequence_error"],
                "No NAK/sequence errors", "nak_tx=0 & nak_rx=0 & frame_sequence_error=0 required"),
        # PFC propagation: Tx-side PFC must be > 0 on queue 3
        make_check(pfc_fail_expr, [pfc_col] if pfc_col in pfc_tx_df.columns else list(pfc_tx_df.columns),
                f"PFC propagated upstream (DUT Tx PFC{queue} > 0)",
                "DUT Tx PFC must be > 0 — proves DUT forwarded PFC from downstream",
                override_df=pfc_tx_df),
    ]
    assert_queries(flow_df, checks)
    logger.info("*** TC5 PASSED ***")
