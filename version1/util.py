import os
import torch


def getSequence(filepath):
    letters = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
               'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
               'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'UNK': 'X', 'PCA': 'X'}
    with open(filepath, 'r') as file:
        seqres = ''
        flag = 0
        for line in file:
            toks = line.split()
            if toks[0] == 'SEQRES':
                if toks[1] == '1':
                    flag += 1
                if flag > 1:
                    break
                tempseq = toks[4:]
                tempseq = [letters[i] for i in tempseq]
                tempseq = ''.join(tempseq)
                seqres += tempseq
    return seqres


def getXYZ(filepath, length):
    with open(filepath, 'r') as file:
        xyz = []
        for line in file:
            toks = line.split()
            if toks[0] == 'ATOM':
                if toks[2] == 'CA' and len(xyz) < length:
                    xyz.append([float(toks[6]), float(toks[7]), float(toks[8])])
    return xyz


def getCaData(filepath='./data/pdb/', maxlen=140):
    for filename in os.listdir(filepath):
        if filename.endswith('.pdb'):
            seqres = getSequence(filepath + filename)
            xyz = getXYZ(filepath + filename, len(seqres))
            # 小于140的补全
            if len(seqres) < maxlen:
                for i in range(maxlen - len(seqres)):
                    seqres += 'X'
                    xyz.append([0, 0, 0])
            with open('./data/CaResult/' + filename + '.txt', 'w') as file:
                # 只取前140个
                for i in range(maxlen):
                    file.write(seqres[i] + ' ' + str(xyz[i][0]) + ' ' + str(xyz[i][1]) + ' ' + str(xyz[i][2]) + '\n')


def getOneHot(filepath):
    with open(filepath, 'r') as file:
        seqres = []
        xyz = []
        for line in file:
            toks = line.split()
            seqres.append(toks[0])
            xyz.append([float(toks[1]), float(toks[2]), float(toks[3])])
    letters = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    seqres = [[1 if i == j else 0 for i in letters] for j in seqres]
    return seqres, xyz


def getTrainData(filepath='./data/CaResult/'):
    trainData = []
    trainLabel = []
    for filename in os.listdir(filepath):
        if filename.endswith('.txt'):
            onehot, xyz = getOneHot(filepath + filename)
            trainData.append(onehot)
            trainLabel.append(xyz)

    trainData = torch.tensor(trainData)
    trainLabel = torch.tensor(trainLabel)
    return trainData, trainLabel


def utilmain():
    # getCaData()
    trainData, trainLabel = getTrainData()
    print(trainData.shape)
    print(trainLabel.shape)
    return trainData, trainLabel
