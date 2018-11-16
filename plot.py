import matplotlib.pyplot as plt
import sys
import pickle

histories = []

histories.append(pickle.load(open("histories/history_inputS16_LS1", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS16_LS2", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS16_LS3", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS16_LS5", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS32_LS1", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS32_LS2", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS32_LS3", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS32_LS5", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS64_LS1", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS64_LS2", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS64_LS3", "rb" ) ))
histories.append(pickle.load(open("histories/history_inputS64_LS5", "rb" ) ))

# summarize history for accuracy
plt.figure()
plt.subplot(121)
for i in range(len(histories)):
    plt.plot(histories[i]['val_acc'])
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['16 notes, LS 0.1', '16 notes, LS 0.2', '16 notes, LS 0.3', '16 notes, LS 0.5','32 notes, LS 0.1', '32 notes, LS 0.2', '32 notes, LS 0.3', '32 notes, LS 0.5','64 notes, LS 0.1', '64 notes, LS 0.2', '64 notes, LS 0.3', '64 notes, LS 0.5'], loc='upper left',bbox_to_anchor=(-0.33, 1.0))

# summarize history for loss
plt.subplot(122)
for i in range(len(histories)):
    plt.plot(histories[i]['val_loss'])
plt.title('validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['16 notes, LS 0.1', '16 notes, LS 0.2', '16 notes, LS 0.3', '16 notes, LS 0.5','32 notes, LS 0.1', '32 notes, LS 0.2', '32 notes, LS 0.3', '32 notes, LS 0.5','64 notes, LS 0.1', '64 notes, LS 0.2', '64 notes, LS 0.3', '64 notes, LS 0.5'], loc='upper right',bbox_to_anchor=(1.27, 1.0))
plt.show()