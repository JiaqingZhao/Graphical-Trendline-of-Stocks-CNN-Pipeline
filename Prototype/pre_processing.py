import cv2
import os

def image_processing(raw_data,data_path,height,width):

    """
    :param raw_data: path where raw data is stored
    :param data_path: path where pre-processed images will be stored
    :param height: new height of images
    :param width: new width of images
    :return:
    """

    class_labels=[]
    category_count=0
    for i in os.walk(raw_data):
        print(i)
        if len(i[2])>1:
            counter=0
            images=i[2]
            class_name=i[0].strip('/')
            path=os.path.join(data_path,class_labels[category_count])
            print("class",class_name)
            for image in images:
                im = cv2.imread(class_name+'/'+image)
                im = cv2.resize(im,(height,width))
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path,str(counter)+'.jpg'),im)
                counter+=1
            category_count+=1

        else:

            number_of_classes=len(i[1])
            print(number_of_classes,i[1])
            class_labels=i[1][:]

if __name__=='__main__':
    height = 100
    width = 100
    raw_data = 'output'
    data_path = 'processed_data'
    if not os.path.exists(data_path):
        for i in ["/train","/test"]:
            image_processing(raw_data + i, data_path, height, width)


