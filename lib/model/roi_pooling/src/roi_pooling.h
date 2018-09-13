int roi_pooling_forward(int pooled_height, int pooled_width, float spatial_scale,
                        THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output);
int roi_pooling_backward(int pooled_height, int pooled_width, float spatial_scale,
                        THFloatTensor * top_grad, THFloatTensor * rois, THFloatTensor * bottom_grad, THIntTensor * argmax);
