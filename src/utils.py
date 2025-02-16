
def extract_network_data(data):
    
    nodes = {node['name']: node for node in data['nodes']}
    links = data['links/ports']
    streams = data['streams']
    return nodes, links, streams
