with open('/home/linzhe/data/flickr30k/results_20130124.token', 'r') as f:
    data = [line.strip('\n').lower().split('\t') for line in f.readlines()]
with open('/home/linzhe/data/flickr30k/flickr.val', 'r') as f:
    val = set([line.strip('\n').lower() for line in f.readlines()])

output = []
for line in data:
    if line[1] not in val:
        output.append((line[0][:-2] + '\n', line[1] + '\n'))
with open('/home/linzhe/data/flickr30k/img.all', 'w') as f:
    for line in output:
        f.write(line[0])  
with open('/home/linzhe/data/flickr30k/flickr.all', 'w') as f:
    for line in output:
        f.write(line[1])



# process_data('/home/linzhe/data/flickr30k/results_20130124.token', '/home/linzhe/baseline/data/flickr/flickr30k.src', '/home/linzhe/baseline/data/flickr/flickr30k.tgt')