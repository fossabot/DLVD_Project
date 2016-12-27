import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import numpy as np

synset = [l.strip() for l in open('/Users/Nutzer1/Downloads/synset.txt').readlines()]

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
  # load image
  img = skimage.io.imread(path)
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img

# returns the top1 string
def print_prob(prob):
  #print prob
  print "prob shape", prob.shape
  pred = np.argsort(prob)[::-1]

  # Get top1 label
  top1 = synset[pred[0]]
  print "Top1: ", top1
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  print "Top5: ", top5
  return top1

with open("/Users/Nutzer1/Downloads/vgg.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={ "images": images })
print "graph loaded from disk"

graph = tf.get_default_graph()

cat = load_image("/Users/Nutzer1/Downloads/elch.jpg")

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  summary_writer = tf.train.SummaryWriter('/Users/Nutzer1/Tensorflow', graph_def=sess.graph_def)
  print "variables initialized"

  batch = cat.reshape((1, 224, 224, 3))
  assert batch.shape == (1, 224, 224, 3)

  feed_dict = { images: batch }

  prob_tensor = graph.get_tensor_by_name("import/prob:0")
  prob = sess.run(prob_tensor, feed_dict=feed_dict)

print_prob(prob[0])