import numpy as np
import matplotlib.pyplot as plt

from imageSegmentation import imageSegmentation

# Test the value of c, report time gain and labels
# r =

image_base_path = r'C:\Users\Gebruiker\PycharmProjects\CV\Assignment_1_Segmentation\image'
def experiment1(r, t):
    time_array = np.zeros((3, 5))
    peak_array = np.zeros((3, 5))
    for image_number in range(1, 4):
        image_path = image_base_path + str(image_number) + '.jpg'
        for i, c in enumerate([1, 2, 3, 4, 5]):
            run = imageSegmentation(image_path, r, t, c, False, 'exp1_im' + str(image_number) + '_c_' + str(c))
            run.run_opt()
            time_array[image_number - 1][i] = run.time_taken
            peak_array[image_number - 1][i] = run.peaks.shape[0]

    with open('exp_c.npy', 'wb') as f:
        np.save(f, time_array)
        np.save(f, peak_array)


# Test the value of r, report time gain and labels
def experiment2(c, t):
    time_array = np.zeros((3, 5))
    peak_array = np.zeros((3, 5))
    for image_number in range(1, 4):
        image_path = image_base_path + str(image_number) + '.jpg'
        for i, r in enumerate([0.05, 0.1, 0.15, 0.2, 0.25]):
            run = imageSegmentation(image_path, r, t, c, True, 'exp1_im' + str(image_number) + '_r_' + str(r))
            run.run_opt()
            time_array[image_number - 1][i] = run.time_taken
            peak_array[image_number - 1][i] = run.peaks.shape[0]

    with open('exp_r.npy', 'wb') as f:
        np.save(f, time_array)
        np.save(f, peak_array)
    # with open('test.npy', 'rb') as f:
    #     a = np.load(f)
    #     b = np.load(f)

# Test basic vs optimized times
def experiment3(r, c, t):
    time_array = np.zeros((3, 2))
    peak_array = np.zeros((3, 2))
    for image_number in range(1, 4):
        image_path = image_base_path + str(image_number) + '.jpg'
        run = imageSegmentation(image_path, r, t, c, True, 'exp1_fast' + str(1))
        run.run_opt()
        time_array[image_number - 1][i] = run.time_taken
        peak_array[image_number - 1][i] = run.peaks.shape[0]

    with open('test.npy', 'wb') as f:
        np.save(f, time_array)
        np.save(f, peak_array)

# Test 3 feature matrix vs 5 feature matrix.
def experiment4():
    pass

def plot(peaks):
    labels = ['0.05', '0.1', '0.15', '0.2', '0.25']

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    print(peaks[0][:])
    rects1 = ax.bar(x - width, peaks[0][:], width, label='London')
    rects2 = ax.bar(x, peaks[1][:], width, label='Japan')
    rects3 = ax.bar(x + width, peaks[2][:], width, label='Woman')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time taken')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()

experiment2(4, 0.001)
# experiment1(0.1, 0.001)
#
# with open('exp_c.npy', 'rb') as f:
#     a = np.load(f)
#     a = np.around(a, decimals=2)
#     b = np.load(f)
#
# plot(a)