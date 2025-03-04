|--- REMDME.md
|--- requirements.txt
|--- setup.py
|--- docs/
|--- tests/
|--- src/
|  |--- __init__.py
|  |--- data/
|  |  |--- input/xx.json
|  |  |___ output/xx.json
|  |---data_handler/
|  |  |--- load_data.py
|  |  |___ save_data.py
|  |--- graph/
|  |  |--- __init__.py
|  |  |___ network_topo.py  # 包含build_graph，draw_topology
|  |--- sorting/
|  |  |--- 
|  |  |___ sort_streams.py # 包含 sort_streams
|  |--- routing/
|  |  |--- __init__.py
|  |  |--- find_k_shortest_path.py
|  |  |--- optimal_routing_algorithm.py # 包含：（select_optimal_routes，get_stream_phy_delay_on_path）
|  |  |___ map_links_to_ports.py  # 包含：（build_link_mapping，convert_stream_routes）
|  |--- scheduling/
|  |  |--- compute_scheduling_params.py  # 包含：（calculate_tdi_and_ts，calculate_tsai_and_ntstc）
|  |  |--- allocate_time_slot.py # 实现函数allocate_time_slots
|  |  |___ derive_gcl.py # 实现函数derive_gcl
|___ main.py  # 主入口程序，调用各个模块实现完整流程
