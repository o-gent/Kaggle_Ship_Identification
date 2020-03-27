import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()


def live_plot(data_point, index):
    """
    plots live data
    """
    plt.scatter(index, data_point)
    plt.pause(0.000001)