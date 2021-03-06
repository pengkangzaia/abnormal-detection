from algorithm.ae import *
from data.SWaT import swat_processor
from utils.eval_methods import *
from utils.utils import *

swat_data = swat_processor.SWaT(batch_size=1000, window_size=12, read_rows=None)
train_loader, val_loader, test_loader = swat_data.get_dataloader()
w_size = swat_data.window_size * swat_data.input_feature_dim
z_size = swat_data.window_size * 5

model = AE(w_size=w_size, z_size=z_size)

is_train = False
if is_train:
    device = get_default_device()
    model = to_device(model, device)
    val_loss_history = training(epochs=30, model=model, train_loader=train_loader, val_loader=val_loader,
                                opt_func=torch.optim.Adam)
    plot_val_history(val_loss_history)
else:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('D://project//work//abnormal-detection//saved_model//ae//ae_swat'))
    else:
        model.load_state_dict(torch.load('D://project//work//abnormal-detection//saved_model//ae//ae_swat', map_location=torch.device('cpu')))

pred_error = testing(model=model, test_loader=test_loader)
bf_search(score=pred_error, label=np.array(swat_data.attack_labels), start=pred_error.min(), end=pred_error.max(),
          step_num=1000,
          display_freq=50)
