import config as conf
import utils as utils

def create_saver(tf, sess) :
    print("create saver")
    saver = tf.train.Saver(tf.global_variables())
    print('Done')
    return saver


def load_gen_last_checkpoint(tf, sess, saver,  path=""):
    print("load last checkpoint")
    c_path = conf.project_path + conf.checkpoints_path + path
    print(c_path)

    try:
        saver.restore(sess, tf.train.latest_checkpoint(c_path))
    except SystemError:
        print("No checkpoint found.")

    print("DONE")


def save_gen_checkpoint(sess, saver, path="", name="\\checkpoint.data"):
    print('save checkpoint')
    utils.make_sure_path_exists(conf.project_path + conf.checkpoints_path + path)
    saver.save(sess, conf.project_path + conf.checkpoints_path + path + name)
    print('Done')
