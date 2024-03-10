import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import argparse
from losses import ContentLoss,StyleLoss

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def parse_args():
    parser = argparse.ArgumentParser(description="args parser for neural style transfer")
    parser.add_argument("--content",type=str,default="data/dancing.jpg", help="content img")
    parser.add_argument("--style",type=str,default="data/picasso.jpg", help="style img")
    parser.add_argument("--imsize",type=int,default=1024,help="img size")
    parser.add_argument("--step",type=int,default=100,help="epoch")
    parser.add_argument("--style_weight",type=float,default=1e5,help="weight of style loss")
    parser.add_argument("--content_weight",type=float,default=2,help="weight of content loss")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # desired size of the output image
    imsize = args.imsize if torch.cuda.is_available() else 128  # use small size if no GPU

    style_img = image_loader(args.style,imsize=imsize,device=device)
    content_img = image_loader(args.content,imsize=imsize,device=device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    input_img = content_img.clone()

    fig, ax = plt.subplots(1,3, figsize=(15,9))

    imshow(ax[0],style_img, title='Style Image')
    imshow(ax[1],content_img, title='Content Image')

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = [0.485, 0.456, 0.406]
    cnn_normalization_std = [0.229, 0.224, 0.225]

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img,
                            device,num_steps=args.step,style_weight=args.style_weight,content_weight=args.content_weight)

    imshow(ax[2],output, title='Output Image')
    plt.ioff()
    plt.show()
    
def image_loader(image_name, imsize:int, device):
    transform = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(ax,tensor, title=None):
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        ax.imshow(image)
        if title is not None:
            ax.set_title(title)
        
def normalizeImg(img, mean, std, device):
    """normalize img value of three channel"""
    img = img.to(device=device)
    if isinstance(mean, list):
        mean = torch.tensor(mean).view(1,-1,1,1).to(device=device)
    else:
        mean.view(1,-1,1,1).to(device=device)
    if isinstance(std, list):
        std = torch.tensor(std).view(1,-1,1,1).to(device=device)
    else:
        std.view(1,-1,1,1).to(device=device)
    normalize_img = (img - mean) / std
    return normalize_img
    
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,device,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    # normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()
    content_img = normalizeImg(content_img,normalization_mean,normalization_std,device)
    style_img = normalizeImg(style_img,normalization_mean,normalization_std,device)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer.to(device=device))

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target).to(device=device)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature).to(device=device)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, device, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img,device)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            normalize_img = normalizeImg(input_img,normalization_mean,normalization_std,device)
            optimizer.zero_grad()
            model(normalize_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == "__main__":
     main()


