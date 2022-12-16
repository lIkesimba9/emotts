
import torch
import os

def main() -> None:

    device = torch.device('cpu')
    
    #model = torch.load('checkpoints/fastspeech2/fastspeech2/1_fastspeech2_model.pth', map_location=device)
    #model = torch.load('checkpoints/fastspeech2_reduced_va/fastspeech2/1_fastspeech2_model.pth', map_location=device)
    #model = torch.load('checkpoints/nat_inflated/feature/300_feature_model.pth', map_location=device)
    #model = torch.load('checkpoints/nat_va_before_duration/feature/1_feature_model.pth', map_location=device)
    #model = torch.load('checkpoints/nat_va_after_duration/feature/1_feature_model.pth', map_location=device)

    checkpoint_list = [
        'checkpoints/nat_base/feature/1_feature_model.pth',
        'checkpoints/fastspeech2/fastspeech2/fastspeech2_model.pth',
        'checkpoints/fastspeech2_reduced_va/fastspeech2/1_fastspeech2_model.pth',
        'checkpoints/nat_inflated/feature/300_feature_model.pth',
        'checkpoints/nat_va_before_duration/feature/1_feature_model.pth',
        'checkpoints/nat_va_after_duration/feature/1_feature_model.pth'
    ]
    #print(len(checkpoint_list))

    for i in range(len(checkpoint_list)):
        checkpoint_name = checkpoint_list[i]
        get_weights_num(checkpoint_name, device)
        print(os.path.getsize(checkpoint_name))


def get_weights_num(checkpoint_name, device):

    print("\n", checkpoint_name)

    param_dict = {}
    param_req_grad_dict = {}

    model = torch.load(checkpoint_name, map_location=device)
    param_total = sum(p.numel() for p in model.parameters()) 
    param_req_grad = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    #print(directory)
    #print("MODEL OVERALL: Total number of parameters = {}, requiring grad = {}\n".format(param_total, param_req_grad))
    param_dict['total'] = param_total
    param_req_grad_dict['total'] = param_req_grad
        
    
    counter = 0
    #sum_p = 0
    for n,m in model.named_modules():
        
        if '.' not in n and n != '':
        #if True:
        #if 'decoder' in n and n != '':
            counter += 1
            # if counter < 4:
            #     # print(counter)
            #print("n = {}".format(n))
            param_total = sum(p.numel() for p in m.parameters()) 
            param_req_grad = sum(p.numel() for p in m.parameters() if p.requires_grad) 
            #print(n)
            #print("Total number of parameters = {}, requiring grad = {}\n".format(param_total, param_req_grad))
            param_dict[n] = param_total
            param_req_grad_dict[n] = param_req_grad
            #sum_p += param_total
            #print(sum_p)

    #print("there are {} blocks in our model".format(counter))
    print(param_dict)
    

    return param_dict, param_total, param_req_grad


if __name__ == "__main__":
    main()