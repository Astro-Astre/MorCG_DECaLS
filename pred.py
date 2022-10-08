import random
from torch.backends import cudnn
from utils.galaxy_dataset import *
import torch

MODEL_PATH = "/data/renhaoye/model_256_Adam_transfer/model_10.pt"


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def answer_prob(out):
    output = []
    q1 = out[0] + out[1] + out[2]
    q2 = out[3] + out[4]
    q3 = out[5] + out[6]
    q4 = out[7] + out[8] + out[9]
    q5 = out[10] + out[11] + out[12] + out[13] + out[14]
    q6 = out[15] + out[16] + out[17]
    q7 = out[18] + out[19] + out[20]
    q8 = out[21] + out[22] + out[23]
    q9 = out[24] + out[25] + out[26] + out[27] + out[28] + out[29]
    q10 = out[30] + out[31] + out[32] + out[33]

    output.append(
        [out[0] / q1, out[1] / q1, out[2] / q1, out[3] / q2, out[4] / q2, out[5] / q3, out[6] / q3, out[7] / q4,
         out[8] / q4,
         out[9] / q4, out[10] / q5, out[11] / q5, out[12] / q5, out[13] / q5, out[14] / q5, out[15] / q6, out[16] / q6,
         out[17] / q6, out[18] / q7, out[19] / q7, out[20] / q7, out[21] / q8, out[22] / q8, out[23] / q8, out[24] / q9,
         out[25] / q9, out[26] / q9, out[27] / q9, out[28] / q9, out[29] / q9, out[30] / q10, out[31] / q10,
         out[32] / q10, out[33] / q10])
    return np.array(output)


def pred(j, rows, w):
    T = 100
    output_list = []
    # data = load_img("/data/renhaoye/MorCG/dataset/out_decals/scaled/" + rows[i].split(" ")[0])
    data = load_img(rows[j].split(" ")[0])
    x = torch.from_numpy(data)
    for i in range(T):
        # print(answer_prob(model(x.to("cuda:0").unsqueeze(0)).data.cpu().numpy()[0,:]).shape)
        # output_list.append(torch.unsqueeze(torch.Tensor(answer_prob(model(x.to("cuda:0").unsqueeze(0)).data.cpu().numpy()[0, :])), 0))
        output_list.append(
            torch.unsqueeze(torch.Tensor(answer_prob(model(x.to("cuda:0").unsqueeze(0)).data.cpu().numpy()[0, :])),
                            0).numpy())
    # y = model(x.to("cuda:0").unsqueeze(0))
    # print(output_list)
    mean = np.mean(np.array(output_list), axis=0)[0, 0, :]
    variance = np.var(np.array(output_list), axis=0)[0, 0, :]
    w.writelines(str(rows[j].split("\n")[0]))
    # print(str(rows[j].split("\n")[0]))
    for i in range(mean.shape[0]):
        w.writelines(" " + str(mean[i]))
    for i in range(variance.shape[0]):
        w.writelines(" " + str(variance[i]))
    w.writelines("\n")
    # q1 = out[0] + out[1] + out[2]
    # q2 = out[3] + out[4]
    # q3 = out[5] + out[6]
    # q4 = out[7] + out[8] + out[9]
    # q5 = out[10] + out[11] + out[12] + out[13] + out[14]
    # q6 = out[15] + out[16] + out[17]
    # q7 = out[18] + out[19] + out[20]
    # q8 = out[21] + out[22] + out[23]
    # q9 = out[24] + out[25] + out[26] + out[27] + out[28] + out[29]
    # q10 = out[30] + out[31] + out[32] + out[33]
    # w.writelines(str(rows[i].split("\n")[0]) + " " +
    #              str(out[0] / q1) + " " +
    #              str(out[1] / q1) + " " +
    #              str(out[2] / q1) + " " +
    #
    #              str(out[3] / q2) + " " +
    #              str(out[4] / q2) + " " +
    #
    #              str(out[5] / q3) + " " +
    #              str(out[6] / q3) + " " +
    #
    #              str(out[7] / q4) + " " +
    #              str(out[8] / q4) + " " +
    #              str(out[9] / q4) + " " +
    #
    #              str(out[10] / q5) + " " +
    #              str(out[11] / q5) + " " +
    #              str(out[12] / q5) + " " +
    #              str(out[13] / q5) + " " +
    #              str(out[14] / q5) + " " +
    #
    #              str(out[15] / q6) + " " +
    #              str(out[16] / q6) + " " +
    #              str(out[17] / q6) + " " +
    #
    #              str(out[18] / q7) + " " +
    #              str(out[19] / q7) + " " +
    #              str(out[20] / q7) + " " +
    #
    #              str(out[21] / q8) + " " +
    #              str(out[22] / q8) + " " +
    #              str(out[23] / q8) + " " +
    #
    #              str(out[24] / q9) + " " +
    #              str(out[25] / q9) + " " +
    #              str(out[26] / q9) + " " +
    #              str(out[27] / q9) + " " +
    #              str(out[28] / q9) + " " +
    #              str(out[29] / q9) + " " +
    #
    #              str(out[30] / q10) + " " +
    #              str(out[31] / q10) + " " +
    #              str(out[32] / q10) + " " +
    #              str(out[33] / q10) + "\n")


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == '__main__':
    init_rand_seed(1926)
    # out_decals = os.listdir("/data/renhaoye/MorCG/dataset/out_decals/scaled/")
    testset = []
    with open("/data/renhaoye/mw_overlap_test.txt", "r") as r:
    # with open("/data/renhaoye/mw_test.txt", "r") as r:
        testsets = r.readlines()
    for i in range(len(testsets)):
        testset.append(testsets[i].split(" label")[0])
        # testset.append(testsets[i].split("\n")[0])
    # print(testset[0].split(" ")[0])
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    model.to("cuda:0")
    model.eval()
    enable_dropout(model)
    for param in model.parameters():
        param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = False
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    name = "overlap_test_mw.txt"

    with open("/data/renhaoye/%s" % name, "w+") as w:
        w.writelines("loc "
                     "smooth_or_featured_smooth smooth_or_featured_featured_or_disk smooth_or_featured_artifact "
                     "disk_edge_on_yes disk_edge_on_no "
                     "has_spiral_arms_yes has_spiral_arms_no "
                     "bar_strong bar_weak bar_no "
                     "bulge_size_dominant bulge_size_large bulge_size_moderate bulge_size_small bulge_size_none "
                     "how_rounded_round how_rounded_medium how_rounded_loose "
                     "edge_on_bulge_boxy edge_on_bulge_none edge_on_bulge_rounded "
                     "spiral_winding_tight spiral_winding_medium spiral_winding_loose "
                     "spiral_arm_count_1 spiral_arm_count_2 spiral_arm_count_3 spiral_arm_count_4 "
                     "spiral_arm_count_more_than_4 spiral_arm_count_cant_tell "
                     "merging_none merging_minor_disturbance merging_major_disturbance merging_merger "

                     "smooth_or_featured_smooth_v smooth_or_featured_featured_or_disk_v smooth_or_featured_artifact_v "
                     "disk_edge_on_yes_v disk_edge_on_no_v "
                     "has_spiral_arms_yes_v has_spiral_arms_no_v "
                     "bar_strong_v bar_weak_v bar_no_v "
                     "bulge_size_dominant_v bulge_size_large_v bulge_size_moderate_v bulge_size_small_v bulge_size_none_v "
                     "how_rounded_round_v how_rounded_medium_v how_rounded_loose_v "
                     "edge_on_bulge_boxy_v edge_on_bulge_none_v edge_on_bulge_rounded_v "
                     "spiral_winding_tight_v spiral_winding_medium_v spiral_winding_loose_v "
                     "spiral_arm_count_1_v spiral_arm_count_2_v spiral_arm_count_3_v spiral_arm_count_4_v "
                     "spiral_arm_count_more_than_4_v spiral_arm_count_cant_tell_v "
                     "merging_none_v merging_minor_disturbance_v merging_major_disturbance_v merging_merger_v\n")
    w = open("/data/renhaoye/%s" % name, "a")
    for j in range(len(testset)):
        # for i in tqdm(range(1)):
        pred(j, testset, w)
    # index = []
    # for idx in range(len(testset)):
    #     index.append(idx)
    # # pred(index[0], testset, w)
    # p = multiprocessing.Pool(20)
    # p.map(partial(pred, rows=testset, w=w), index)
    # p.close()
    # p.join()
    w.close()
