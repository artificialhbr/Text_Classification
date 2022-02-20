import torch
from importlib import import_module
import argparse
import pickle as pkl
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--embed',default='embedding_SougouNews.npz',type=str,help='embedding_SougouNews.npz or embedding_Tencent.npz')
args = parser.parse_args()
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

if __name__ == '__main__':
    dataset = 'text/THUCNews'  # 数据集
    embedding = args.embed
    model_name = args.model
    x = import_module('models.' + model_name)
    x.Config(dataset, embedding)
    config = x.Config(dataset, embedding)
    model = x.Model(config)
    model.load_state_dict(torch.load(config.save_path))
    model = model.to(config.device)
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    content = ["iOS 15某功能抄袭锤子，罗永浩亲自回应了","TCL旗下摩迅半导体AI芯片研发项目落地临港新片区","NBA30天30队之国王篇：福克斯率队刮起青春风暴"
    ,"许昕把全运会金牌送给二胎女儿：妹妹的见面礼已到达","渣叔头大！利物浦遭升班马3闷棍：阿诺德两次被完爆+4人眼神防守","上海楼市调查：价格虚高学区房挤掉300万水分 投资客淡出"]
    words_line = []
    tokenizer = lambda x: [y for y in x]
    pad_size = config.pad_size
    for item in content:
        token = tokenizer(item)
        seq_len = len(token)
        if seq_len<pad_size:
            token.extend([vocab.get(PAD)]*(pad_size-seq_len))
        else:
            token = token[:pad_size]
        arr = []
        for i in token:
            arr.append(vocab.get(i,vocab.get(UNK)))
        words_line.append((arr,seq_len))
    x = torch.LongTensor([_[0] for _ in words_line]).to(config.device)
    seq = torch.LongTensor([_[1] for _ in words_line]).to(config.device)
    train_set = (x,seq)
    outputs = model(train_set)
    index = torch.max(outputs,1)[1]
    brr = []
    with open(dataset+'/data/class.txt')as f:
        for i in f.readlines():
            brr.append(i.strip())
    for i in range(len(content)):
        print('{0}:{1}'.format(content[i],brr[index[i]]))      
        
