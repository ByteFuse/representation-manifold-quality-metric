import torch

def return_fgsm_contrastive_attack_images(
    images,
    model,
    loss_fn,
    val_transform,
    transform,
    iterations=1,
    device='cuda',
    epsilon=0):

    images = images.clone().detach().to(device)
    adv_images = images.clone().detach()
    
    for _ in range(iterations):
        
        ref_images = transform(images)
        adv_images.requires_grad = True
        
        ref_emb, query_emb = model(ref_images), model(val_transform(adv_images))
        loss = loss_fn(ref_emb, query_emb).to(device)
        
        grad = torch.autograd.grad(
            loss, adv_images,
            retain_graph=False,
            create_graph=False
        )[0]
        
        adv_images = adv_images.detach() + epsilon*grad.sign()
        delta = torch.clamp(adv_images - images, min=-0.3, max=0.3)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    
    return val_transform(adv_images)


def return_fgsm_supervised_attack_images(
    images,
    labels,
    model,
    loss_fn,
    final_transform,
    iterations=1,
    require_logits=False,
    device='cuda',
    epsilon=0
):
        
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_images = images.clone().detach()

    for _ in range(iterations):
        adv_images.requires_grad = True
        
        if require_logits:
            embeddings, logits = model(final_transform(adv_images))
            loss = loss_fn(embeddings, logits, labels).to(device)
        else:
            embeddings = model(adv_images)
            loss = loss_fn(embeddings, labels).to(device)
            
        grad = torch.autograd.grad(
            loss, adv_images,
            retain_graph=False,
            create_graph=False
        )[0]

        adv_images = adv_images.detach() + epsilon*grad.sign()
        delta = torch.clamp(adv_images - images, min=-0.3, max=0.3)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return final_transform(adv_images)