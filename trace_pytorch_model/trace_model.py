import torch
from model_to_trace import HomoNet_ICSTN_Down_stu


device = 'cpu' # 'cpu' 'cuda:0'
img_height = 224
img_width = 320
stu_model_path = "UAHN_fcdrop05_16.pth.tar"
show_photometric_error = True


def main():

    pretrained_stu_model_data = torch.load(stu_model_path, map_location=torch.device('cpu'))
    print("Pre-trained model '{}'".format(stu_model_path))
    HomoNet_model_stu = HomoNet_ICSTN_Down_stu(img_height, img_width, device, pretrained_stu_model=pretrained_stu_model_data, dropout_rate=0.05, show_photometric_error=show_photometric_error).to(device)
    print("Trace the model for cpp! With PyTorch version", torch.__version__)
    print("Target device for traced cpp model deployment is", device)
    # An example input you would normally provide to your model's forward() method.
    img1 = torch.ones(1, 1, 224, 320)*0.2
    img2 = torch.ones(1, 1, 224, 320)*0.5
    homo8 = torch.ones(1, 1, 4, 2)*0.9
    
    HomoNet_model_stu.eval()

    for p in HomoNet_model_stu.parameters():
        p.requires_grad = False # NOTE without it, the dropout and ensemble have GPU RAM increasing with time when running the traced model in cpp
     
    # mean, var = HomoNet_model_stu(img1.to(device),img2.to(device), homo8.to(device))
    # mean, var = HomoNet_model_stu(img1.to(device),img2.to(device))
    # print(mean.size(), var.size())

    with torch.no_grad():

        # full model # check_trace=False to avoid the warnings caused by dropout
        traced_script_module = torch.jit.trace(HomoNet_model_stu, (img1.to(device), img2.to(device)), check_trace=False) # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        
        # when using prior (run fewer than four network blocks)
        traced_script_module_prior = torch.jit.trace(HomoNet_model_stu, (img1.to(device), img2.to(device), homo8.to(device)), check_trace=False) # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        
        if show_photometric_error:
            traced_script_module.save("traced_full_model_showError.pt")
            traced_script_module_prior.save("traced_model_3_blocks_using_prior_showError.pt")
        else:
            traced_script_module.save("traced_full_model.pt")
            traced_script_module_prior.save("traced_model_3_blocks_using_prior.pt")


if __name__ == '__main__':
    main()