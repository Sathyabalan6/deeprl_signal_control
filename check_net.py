import xml.etree.ElementTree as ET

tree = ET.parse('custom_net/data/in/custom.net.xml')
root = tree.getroot()
print('root:', root.tag)

TLS = {
    '5460773146', '5528559489', '5528559490',
    'cluster_13067792373_13067792374_1645179383',
    'cluster_2212827519_2212827520_268789820',
    'cluster_2212827553_246824741'
}

# Find TLS junctions
tls_found = []
for j in root.findall('junction'):
    if j.get('type') == 'traffic_light':
        tls_found.append(j.get('id'))
print('TLS junctions found:', tls_found)

# Find edges touching TLS
touching = []
for edge in root.findall('edge'):
    eid = edge.get('id', '')
    if eid.startswith(':'):
        continue
    frm = edge.get('from', '')
    to = edge.get('to', '')
    if frm in TLS or to in TLS:
        lanes = edge.findall('lane')
        allows = [l.get('allow', 'ALL') for l in lanes]
        touching.append((eid, frm, to, allows))

print('\nEdges touching TLS (%d):' % len(touching))
for eid, frm, to, allows in touching:
    print(' ', eid, frm, '->', to, allows)

# Find valid car edges (no allow restriction = open to all)
car_edges = []
for edge in root.findall('edge'):
    eid = edge.get('id', '')
    if eid.startswith(':'):
        continue
    lanes = edge.findall('lane')
    if any(l.get('allow', '') == '' for l in lanes):
        car_edges.append(eid)

print('\nTotal car edges:', len(car_edges))
print('First 10:', car_edges[:10])
