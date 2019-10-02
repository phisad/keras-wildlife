'''

NOT USED FOR NOW

Created on 11.05.2019

@author: Philipp
'''


def __file_number(image_file):
    filename = image_file.split("\\")[-1]
    filename = filename.split(".")[0]
    filenumber = filename.split("SUNP")[1]
    return int(filenumber)


def get_sequences(images_list):
    sequences = []
    sequence = []
    for idx, (image, labels) in enumerate(images_list):
        if idx == 0:
            sequence.append((image, labels))
            continue

        # prev_image, prev_image_labels = images_list[idx - 1]
        prev_image, _ = images_list[idx - 1]
        seq_num = __file_number(image)
        prev_seq_num = __file_number(prev_image)
        if seq_num - 1 != prev_seq_num:
            sequences.append(sequence)
            sequence = []

        sequence.append((image, labels))

    print("Sequences: {}".format(len(sequences)))
    return sequences
