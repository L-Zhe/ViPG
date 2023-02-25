from    nltk.translate.bleu_score import corpus_bleu

class Eval:

    def __init__(self, reference_file):
        self.reference = self.read_file(reference_file, reference=True)

    def read_file(self, file, reference=False):
        with open(file, 'r') as f:
            if reference:
                data = [[seq.strip('\n').lower().split() for seq in line.strip('\n').split('\t')]
                          for line in f.readlines()]
            else:
                data = [[word.lower() for word in line.strip('\n').split()] for line in f.readlines()]
        return data

    def __call__(self, candidate_file):
        candidate = self.read_file(candidate_file, reference=False)
        bleu4 = corpus_bleu(self.reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)) * 100
        print(corpus_bleu(self.reference, candidate, weights=(1, 0, 0, 0)) * 100)
        print(corpus_bleu(self.reference, candidate, weights=(0, 1, 0, 0)) * 100)
        print(corpus_bleu(self.reference, candidate, weights=(0, 0, 1, 0)) * 100)
        print(corpus_bleu(self.reference, candidate, weights=(0, 0, 0, 1)) * 100)
        print('BLEU4: %.2f' % bleu4)
        return bleu4, str(bleu4)

if __name__ == '__main__':
    source = '/home/linzhe/output/flickr_output/flickr.src'
    target = '/home/linzhe/output/flickr_output/our_one_caption_google.flickr'
    # source = '/home/linzhe/output/mscoco_output/mscoco.src'
    # target = '/home/linzhe/output/mscoco_output/our_one_caption_google.mscoco'
    eval = Eval(source)
    eval(target)
