
import pandas as pd
from matplotlib import pyplot as plt

def plot(train_df, valid_df, epoch_range_train, epoch_range_valid):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    # 第一行：分类相关的指标（Cls_Loss, Precision, Recall, F1）
    axs[0, 0].plot(train_df['Epoch'][0: epoch_range_train].to_numpy(), train_df['Cls_Loss'][0: epoch_range_train].to_numpy(), label='Train Cls Loss', marker='o', color='blue')
    axs[0, 0].plot(valid_df['Epoch'][0: epoch_range_valid].to_numpy(), valid_df['Cls_Loss'][0: epoch_range_valid].to_numpy(), label='Valid Cls Loss', marker='x', color='red')
    axs[0, 0].set_ylim([0, 2])
    axs[0, 1].plot(train_df['Epoch'][0: epoch_range_train].to_numpy(), train_df['Precision'][0: epoch_range_train].to_numpy(), label='Train Precision', marker='o', color='blue')
    axs[0, 1].plot(valid_df['Epoch'][0: epoch_range_valid].to_numpy(), valid_df['Precision'][0: epoch_range_valid].to_numpy(), label='Valid Precision', marker='x', color='red')
    axs[0, 1].set_ylim([0, 1])
    axs[0, 2].plot(train_df['Epoch'][0: epoch_range_train].to_numpy(), train_df['Recall'][0: epoch_range_train].to_numpy(), label='Train Recall', marker='o', color='blue')
    axs[0, 2].plot(valid_df['Epoch'][0: epoch_range_valid].to_numpy(), valid_df['Recall'][0: epoch_range_valid].to_numpy(), label='Valid Recall', marker='x', color='red')
    axs[0, 2].set_ylim([0, 1])

    # 第二行：定位相关的指标（F1, BD_Loss, BD_Ciou）
    axs[1, 0].plot(train_df['Epoch'][0: epoch_range_train].to_numpy(), train_df['F1'][0: epoch_range_train].to_numpy(), label='Train F1', marker='o', color='blue')
    axs[1, 0].plot(valid_df['Epoch'][0: epoch_range_valid].to_numpy(), valid_df['F1'][0: epoch_range_valid].to_numpy(), label='Valid F1', marker='x', color='red')
    axs[1, 0].set_ylim([0, 1])
    axs[1, 1].plot(train_df['Epoch'][0: epoch_range_train].to_numpy(), train_df['BD_Loss'][0: epoch_range_train].to_numpy(), label='Train BD Loss', marker='o', color='blue')
    axs[1, 1].plot(valid_df['Epoch'][0: epoch_range_valid].to_numpy(), valid_df['BD_Loss'][0: epoch_range_valid].to_numpy(), label='Valid BD Loss', marker='x', color='red')
    axs[1, 1].set_ylim([0, 0.03])
    axs[1, 2].plot(train_df['Epoch'][0: epoch_range_train].to_numpy(), train_df['BD_ciou'][0: epoch_range_train].to_numpy(), label='Train BD ciou', marker='o', color='blue')
    axs[1, 2].plot(valid_df['Epoch'][0: epoch_range_valid].to_numpy(), valid_df['BD_Ciou'][0: epoch_range_valid].to_numpy(), label='Valid BD ciou', marker='x', color='red')
    axs[1, 2].set_ylim([-1, 1])

    # 设置图例和标题
    for ax in axs.flat:
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')

    axs[0, 0].set_title('Classification Loss')
    axs[0, 1].set_title('Precision')
    axs[0, 2].set_title('Recall')
    axs[1, 0].set_title('F1 Score')
    axs[1, 1].set_title('Bounding Box Loss')
    axs[1, 2].set_title('Bounding Box CIOU')

    plt.tight_layout()
    plt.show()


def main():
    train_df = pd.read_csv('./runs/2/train_epoch.txt')
    valid_df = pd.read_csv('./runs/2/valid_epoch.txt')
    plot(train_df, valid_df, 100, 100)


if __name__ == '__main__':
    main()


# import pandas as pd
# import proplot as pplt
#
# def plot(train_df, valid_df, epoch_range_train, epoch_range_valid):
#     # 使用ProPlot创建图和子图网格
#     fig, axs = pplt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=False, sharey=False)
#
#     # 第一行：分类相关的指标（Cls_Loss, Precision, Recall）
#     axs[0].plot(train_df['Epoch'][0: epoch_range_train], train_df['Cls_Loss'][0: epoch_range_train], label='Train Cls Loss', marker='o', color='blue')
#     axs[0].plot(valid_df['Epoch'][0: epoch_range_valid], valid_df['Cls_Loss'][0: epoch_range_valid], label='Valid Cls Loss', marker='x', color='red')
#     axs[0].set_ylim([0, 2])
#
#     axs[1].plot(train_df['Epoch'][0: epoch_range_train], train_df['Precision'][0: epoch_range_train], label='Train Precision', marker='o', color='blue')
#     axs[1].plot(valid_df['Epoch'][0: epoch_range_valid], valid_df['Precision'][0: epoch_range_valid], label='Valid Precision', marker='x', color='red')
#     axs[1].set_ylim([0, 1])
#
#     axs[2].plot(train_df['Epoch'][0: epoch_range_train], train_df['Recall'][0: epoch_range_train], label='Train Recall', marker='o', color='blue')
#     axs[2].plot(valid_df['Epoch'][0: epoch_range_valid], valid_df['Recall'][0: epoch_range_valid], label='Valid Recall', marker='x', color='red')
#     axs[2].set_ylim([0, 1])
#
#     # 第二行：定位相关的指标（F1, BD_Loss, BD_Ciou）
#     axs[3].plot(train_df['Epoch'][0: epoch_range_train], train_df['F1'][0: epoch_range_train], label='Train F1', marker='o', color='blue')
#     axs[3].plot(valid_df['Epoch'][0: epoch_range_valid], valid_df['F1'][0: epoch_range_valid], label='Valid F1', marker='x', color='red')
#     axs[3].set_ylim([0, 1])
#
#     axs[4].plot(train_df['Epoch'][0: epoch_range_train], train_df['BD_Loss'][0: epoch_range_train], label='Train BD Loss', marker='o', color='blue')
#     axs[4].plot(valid_df['Epoch'][0: epoch_range_valid], valid_df['BD_Loss'][0: epoch_range_valid], label='Valid BD Loss', marker='x', color='red')
#     axs[4].set_ylim([0, 0.03])
#
#     axs[5].plot(train_df['Epoch'][0: epoch_range_train], train_df['BD_ciou'][0: epoch_range_train], label='Train BD ciou', marker='o', color='blue')
#     axs[5].plot(valid_df['Epoch'][0: epoch_range_valid], valid_df['BD_Ciou'][0: epoch_range_valid], label='Valid BD ciou', marker='x', color='red')
#     axs[5].set_ylim([-1, 1])
#
#     # 设置图例和标题
#     titles = ['Classification Loss', 'Precision', 'Recall', 'F1 Score', 'Bounding Box Loss', 'Bounding Box CIOU']
#     for ax, title in zip(axs, titles):
#         ax.legend()
#         ax.format(title=title, xlabel='Epoch', ylabel='Value')
#
#     fig.tight_layout()
#     fig.show()
#
# def main():
#     # 假设你已经有了CSV文件的路径
#     train_df = pd.read_csv('./runs/2/train_epoch.txt')
#     valid_df = pd.read_csv('./runs/2/valid_epoch.txt')
#     # 若CSV文件中没有Epoch列为索引，则需要指定epoch_range_train和epoch_range_valid的正确值
#     plot(train_df, valid_df, len(train_df), len(valid_df))
#
# if __name__ == '__main__':
#     main()


