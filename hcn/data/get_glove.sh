#wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip glove.6B.zip glove.6B.100d.txt
DIR=`pwd`
python -c "from gensim.test.utils import datapath; from gensim.scripts.glove2word2vec import glove2word2vec; glove_file = datapath('$DIR/glove.6B.100d.txt'); w2v_file = datapath('$DIR/glove.6B.100d.w2v.txt'); glove2word2vec(glove_file, w2v_file)"
