import tensorflow as tf
import numpy as np
# Need this function to determine the IoU of the bounding boxes with the ground truth
def bbox_intersection_over_union(box1, box2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, w1, h1 = tf.split(bboxes1, 4, axis=1)
    x21, y21, w2, h2 = tf.split(bboxes2, 4, axis=1)

    x12 = w1 + x11
    y12 = h1 + y11
    x22 = w2 + x21
    y22 = h2 + y21
    

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    inter_area = tf.maximum(xI2 - xI1 + 1, tf.constant(0.0)) * tf.maximum((yI2 - yI1 + 1), tf.constant(0.0)) 

    bboxes1_area = (w1 + 1) * (h1 + 1)
    bboxes2_area = (w2 + 1) * (h2 + 1)

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    return tf.transpose(tf.maximum(inter_area / union, 0))


# Input:
# @param bboxes : An Nx4 array of bounding boxes, where N is the number of Rois
# @param score : the computed likeliness score of size N corresponding to the bounding box's estimated
# likeliness to be an object
# @param gt: The set of ground truth labels
def faster_rcnn_loss(bboxes, scores, gt):
    intersection = tf.reduce_max(bbox_intersection_over_union(bboxes, gt), axis=1)
    is_intersect = tf.greater(intersection, tf.constant(0.5))
    reg_loss = tf.where(is_intersect, intersection, tf.zeros_like(intersection))
    _, score_loss = tf.split(scores, 2, axis=1)
    print 'score loss' , score_loss
    print 'intersection', intersection
    score_loss = tf.reduce_max(score_loss, axis=1)
    score_loss = tf.where(is_intersect, -tf.log(score_loss), tf.zeros_like(score_loss))
    return score_loss + 10*reg_loss 

if __name__ == '__main__':
    bboxes1 = tf.placeholder(tf.float32)
    bboxes2 = tf.placeholder(tf.float32)
    overlap_op = bbox_intersection_over_union(bboxes1, bboxes2)

    bboxes1_vals = np.array([[39, 63, 203, 112], [0, 0, 10, 10]], dtype=np.float32)
    bboxes2_vals = np.array([[3, 4, 24, 32], [54, 66, 198, 114], [6, 7, 60, 44], [-10, -10, 20, 20]], dtype=np.float32)

    with tf.Session() as sess:
        overlap = sess.run(overlap_op, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
        })

    print 'Overlap op'
    print(overlap)
    # Let's say box 2 was the predicted, and 1 the ground truth
    scores_vals = np.array([[ 0.4, 0.6], [0.1, 0.9], [0.5, 0.5], [1.0, 0.0]])
    scores = tf.placeholder(tf.float32)

    loss_op = faster_rcnn_loss(bboxes2, scores, bboxes1)

    with tf.Session() as sess:
        loss = sess.run(loss_op, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
	    scores: scores_vals,
        })
  
    print 'loss op'
    print(loss)
