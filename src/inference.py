from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
import argparse
import torch
import sys
from transformers import AutoTokenizer, PretrainedConfig
import math


def main(args):
    def tokenize_prompt(tokenizer, prompt):
        padding_length = 385 # this one should be divisible by tokenizer.model_max_length
        text_inputs = tokenizer(
            text=prompt,
            padding="max_length",
            max_length=padding_length,
            truncation=True,
            return_tensors="pt"
        )

        text_input_ids = text_inputs.input_ids
        return text_input_ids

    def tokenize_neg_prompt(tokenizer, prompt, max_length):
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=False,
            return_tensors="pt"
        )

        text_input_ids = text_inputs.input_ids
        return text_input_ids

    def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
    ):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")


    def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None, neg_prompt=False, prompt_max_length=None):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders): # create a prompt embeds from each text encoders (there are 2 text encoders)
                                                        # each text encoders will encode the same prompt
            
            #print("**********************************************************")
            if tokenizers is not None:
                if neg_prompt:
                    assert prompt_max_length is not None
                    tokenizer = tokenizers[i]
                    text_input_ids = tokenize_neg_prompt(tokenizer, prompt, prompt_max_length)
                else:
                    tokenizer = tokenizers[i]
                    text_input_ids = tokenize_prompt(tokenizer, prompt)
                    
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            count = 0
            concat_embeds = []
            pooled_prompt_embeds = 0
            #print(type(text_input_ids))
            for n in range(0, text_input_ids.shape[-1], max_length):
                

                prompt_embeds = text_encoder(
                    text_input_ids[:, n:n+max_length].to(text_encoder.device), output_hidden_states=True, return_dict=False
                )
                '''
                prompt_embeds is a tuple of 3 elements:
                - the first one at index 0 is a torch tensor that is the pooler_output of shape (batch_size, hidden_size)((batch_size, 1280)
                    in this case)
                - the second one at index 1 is a torch tensor last_hidden_state (the embeddings of each captions)(the final output from the model) of shape (batch_size, sequence_length, hidden_size) ((batch_size, sequence_length, 1280) in this case)
                - the third one at index 2 is a tuple of torch tensor that are hidden_states. each hidden states in this tuple is an output from a layer of the
                    text encoder (the first element in this tuple is the hidden state(embedding) output from the embeddings layer, and second element is the output from the second layer and so on)
                    each element(which is hidden state) is of shape (batch_size, sequence_length, hidden_size) ((batch_size, sequence_length, 1280) in this case)(have the same shape as the second element (last_hidden_state) in the big tuple)
                '''

                # We are only ALWAYS interested in the pooled output of the final text encoder
                if count == 0 or count == 1:
                #if count == 0:
                    pooled_prompt_embeds = pooled_prompt_embeds + prompt_embeds[0]
                else:
                    pooled_prompt_embeds += prompt_embeds[0]

                prompt_embeds = prompt_embeds[-1][-2] # prompt_embeds[-1] return the last element in the tuple(the list of hidden states from each layer)
                                                    # then do [-2] return the second to the last layer’s hidden states. get the second to the last layer not the last because in many models, the final layer is too specialized for the training objective of that model(e.g., masked language modeling in BERT).
                                                    # The second-to-last layer often contains richer, more general representations, making it useful for downstream tasks like text-to-image alignment.
            
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1) # this line can feel like redundant but is used to make sure that the prompt embeddings will have the shape (batch size, sequence length, hidden size) (for example, maybe the text encoder return the embeddings of shape (batch size, sequence length, 1, hidden size) instead)
                concat_embeds.append(prompt_embeds)
                #print(prompt_embeds.shape)
                count += 1

            pooled_prompt_embeds /= count # averaging the pooled embeddings since each time you the pooled embeddings of a pertion of the prompt so you can add all the pooled embedings of each portion together and take the average. 
            prompt_embeds = torch.cat(concat_embeds, dim=1)
            prompt_embeds_list.append(prompt_embeds)
            #print(len(prompt_embeds_list))

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) # combine the embeds of the 2 encoders (the 2 encoder encode the same string)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1) # ensure that the pooled embeds have the shape (batch_size, hidden size) because of the same reason 
        return prompt_embeds, pooled_prompt_embeds


    def compute_time_ids(original_size, crops_coords_top_left):
        target_size = (1152,896) # this is the original one
        #target_size = (0,0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(device, dtype=torch.float16)
        return add_time_ids


    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.load_lora_weights(args.lora_path)
    #pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=False) # improve the inference speed by 20-30% with torch.compile. Simple wrap the unet with torch compile before running the pipeline


    '''
    # script for combinig multiple lora
    pipe.load_lora_weights(
        "/home/public/htnguyen/project/diffusers/rank32-textencoder-const1e-4",
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="quality_lora"
    )

    pipe.load_lora_weights(
        "/home/public/htnguyen/project/diffusers/final-rank128-textencoder",
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="text_title_lora"
    )

    pipe.set_adapters(["quality_lora", "text_title_lora"], adapter_weights=[0.7, 0.3])
    '''

    '''
    prompt = [#'Book Cover - This book title is "speaker for the dead" . This book publisher is "tor books" . This book Genres tags are science fiction , fiction , fantasy , science fiction fantasy , young adult , audiobook , science fiction , aliens , space , novels , space , space opera . now available in mass market the revised definitive edition of the hugo and nebula classic . in this second book in the saga set years after the terrible war ender wiggin is reviled by history as the xenocide the destroyer of the alien buggers . now ender tells the true story of the war and seeks to stop history from repeating itself . in the aftermath of his terrible war ender wiggin disappeared and a powerful voice arose the speaker for the dead who told the true story of the bugger long years later a second alien race has been discovered but again the aliens ways are strange and frightening again humans die . and it is only the speaker for the dead who is also ender wiggin the xenocide who has the courage to confront the mystery and the for the dead the second novel in orson scott card ender quintet is the winner of the nebula award for best novel and the hugo award for best novel .',
            "Book Cover - This book title is \"love and mistletoe\". This book publisher is \"beaverstone press llc\". This book genres are romance , contemporary , romance , contemporary romance , holiday , christmas , holiday , cultural , ireland , novella , romance , m f romance , business , amazon , new adult. an alternate cover edition can be found here . stand alone christmas novella in the ballybeg series of irish romantic comedies . love laughter and a happily ever after during the festive season kissed by christmas true love by new year policeman brian glenn wants a promotion . studying for a degree in criminology is the first step . when a member of ballybeg most notorious family struts into his forensic psychology class his hopes for a peaceful semester vanish . sharon maccarthy is the last woman he should get involved with however hot and bothered she makes him get under his police uniform . can he survive the semester without succumbing to her charms sharon had a rough few months . she knows her future job prospects depend on her finally finishing her degree . when she is paired with her secret crush for the semester project she sees a chance for happiness . can she persuade brian that there is more to her than sequins high heels and a rap sheet.",
            #"Book Cover - This book title is \"stolen songbird\". This book publisher is \"strange chemistry\". This book genres are fantasy , young adult , romance , fantasy , magic , young adult , young adult fantasy , fantasy , paranormal , fantasy , high fantasy , fiction , fairies , fae , paranormal , witches. for five centuries a witch curse has bound the trolls to their city beneath the mountain . when ccile de troyes is kidnapped and taken beneath the mountain she realises that the trolls are relying on her to break the has only one thing on her mind escape . but the trolls are clever fast and inhumanly strong . she will have to bide her timebut the more time she spends with the trolls the more she understands their plight . there is a rebellion brewing . and she just might be the one the trolls were looking for.",
            #"Book Cover - This book title is \"my almost epic summer\". This book publisher is \"putnam juvenile\". This book genres are young adult , young adult , teen , fiction , realistic fiction , audiobook , humor , young adult , coming of age. irene got big dreams someday she will own a salon in where her specialty will be recreating the hairstyles of famous literary heroines . and it is a good thing she has dreams since reality is harsh . she is just been fired from her mom beauty salon for her shampooing technique and is forced to take the only other job she can find babysitting . now she is stuck at the beach entertaining kids while everyone else is having a glamorous summer . will she ever get a life then she meets starla a beautiful lifeguard whose diva attitude dangerous obsessions male admirers and blog supply irene with enough drama and romance to fill a book . amidst the complicated friendships inconvenient crushes and occupational mishaps that seem to define this summer irene suddenly and unexpectedly finds that the countdown to real life is over and her fate is in her hands.",
            "Book Cover - This book title is \"aru shah and the tree of wishes\". This book publisher is \"rick riordan presents\". This book genres are childrens , middle grade , fantasy , fantasy , mythology , young adult , fiction , adventure , audiobook , childrens , fantasy , magic , fantasy , urban fantasy. war between the devas and the demons is imminent and the otherworld is on high alert . when intelligence from the human world reveals that the sleeper is holding a powerful clairvoyant and her sister captive aru and her friends launch a mission . the captives a pair of twins turn out to be the newest pandava sisters though according to a prophecy one sister is not the celebration of holi the heavenly attendants stage a massage pr rebranding campaign to convince everyone that the pandavas are to be trusted . as much as aru relishes the attention she fears that she is destined to bring destruction to her sisters as the sleeper has predicted . aru believes that the only way to prove her reputation is to find the kalpavriksha the tree that came out of the ocean of milk when it was churned . if she can reach it before the sleeper perhaps she can turn everything around with one what you wish for aru .",
            #"This book title is \"till we have faces\". This book publisher is \"harcourt paperbacks\". This book genres are fiction , fantasy , classics , fantasy , mythology , christian , religion , religion , christianity , literature , christian fiction , philosophy. in this timeless tale of two mortal one beautiful and one . lewis reworks the classical myth of cupid and psyche into an enduring piece of contemporary fiction . this is the story of orual psyche embittered and ugly older sister who posessively and harmfully loves psyche . much to orual frustration psyche is loved by cupid the god of love himself setting the troubled orual on a path of moral against the backdrop of glome a barbaric world the struggles between sacred and profane love are illuminated as orual learns that we can not understand the intent of the gods till we have faces and sincerity in our souls and selves.",
            #"Book Cover - This book title is \"a gathering of shadows\". This book publisher is \"tor books\". This book genres are fantasy , young adult , fiction , fantasy , magic , adult , lgbt , audiobook , adventure , romance , fantasy , high fantasy. it has been four months since a mysterious obsidian stone fell into kell possession . four months since his path crossed with delilah bard . four months since prince rhy was wounded and since the nefarious dane twins of white london fell and four months since the stone was cast with holland dying body through the rift back into black restless after having given up his smuggling habit kell is visited by dreams of ominous magical events waking only to think of lila who disappeared from the docks as she always meant to do . as red london finalizes preparations for the element games an extravagant international competition of magic meant to entertain and keep healthy the ties between neighboring countries a certain pirate ship draws closer carrying old friends back into while red london is caught up in the pageantry and thrills of the games another london is coming back to life . after all a shadow that was gone in the night will reappear in the morning . but the balance of magic is ever perilous and for one city to flourish another london must fall.",
            #"Book Cover - This book title is \"breathe into me\". This book publisher is \"amanda stone\". This book genres are new adult , romance , academic , college , contemporary , romance , contemporary romance , sociology , abuse , young adult , adult , womens fiction , chick lit , realistic fiction. eighteen kelsey rien is more than ready to leave her past behind nothing more than to walk into a room without everyone knowing the horrific details of an event that changed her life six years ago she vows to concentrate on school and make something of then she meets kane riley the local bad boy . kane reputation is far from perfect but is there more to him than what everyone else sees kelsey soon learns that you ca not run from your past no matter how hard you try . when her nightmares find her once again kelsey must find the courage to face the demons that have been haunting her and save the people she loves most.",
            #"Book Cover - This book title is \"surrendered\". This book publisher is \"adler and holt\". This book genres are adult fiction , erotica , romance , romance , contemporary romance , romance , romantic suspense , romance , erotic romance , romantic. newly updated version enjoy the newly revised and additionally edited version with a little sneak peek at the end the chemistry between them is undeniable . he captivated her from the first moment she laid eyes on fate brought the interior designer danielle austen and the sexy mogul harrison towers together their pasts continue to threaten their future together . their passion fuels their fire while the couple continues to overcome the hurdles between them . danielle has finally broken through harrison aloof controlling exterior . however his devious jilted marion devereaux continues to threaten danielle safety and seems to have help from someone close to danielle . will harrison and danielle be able to face these challenges together will their love be enough to conquer marion evil plots will they get their storybook ending or will marion have her revenge this book is intended for mature audiences.",
            #"Book Cover - This book title is \"the white giraffe\". This book publisher is \"dial books young readers\". This book genres are fantasy , animals , fiction , childrens , childrens , middle grade , cultural , africa , adventure , mystery , young adult , childrens , juvenile. when martine home in england burns down killing her parents she must go to south africa to live on a wildlife game preserve called sawubona with the grandmother she did not know she had . almost as soon as she arrives martine hears stories about a white giraffe living in the preserve . but her grandmother and others working at sawubona insist that the giraffe is just a myth . martine is not so sure until one stormy night when she looks out her window and locks eyes with jemmy a young giraffe . why is everyone keeping jemmy existence a secret does it have anything to do with the rash of poaching going on at sawubona martine needs all of the courage and smarts she has not to mention a little african magic to find out . children author lauren john brings us deep into the african world where myths become reality and a young girl with a healing gift has the power to save her home and her one true friend.",
            'Book Cover - This book title is "speaker for the dead" . This book publisher is "tor books" . This book Genres tags are science fiction , fiction , fantasy , science fiction fantasy , young adult , audiobook , science fiction , aliens , space , novels , space , space opera . now available in mass market the revised definitive edition of the hugo and nebula classic . in this second book in the saga set years after the terrible war ender wiggin is reviled by history as the xenocide the destroyer of the alien buggers . now ender tells the true story of the war and seeks to stop history from repeating itself . in the aftermath of his terrible war ender wiggin disappeared and a powerful voice arose the speaker for the dead who told the true story of the bugger long years later a second alien race has been discovered but again the aliens ways are strange and frightening again humans die . and it is only the speaker for the dead who is also ender wiggin the xenocide who has the courage to confront the mystery and the for the dead the second novel in orson scott card ender quintet is the winner of the nebula award for best novel and the hugo award for best novel .',
            #'Book Cover - This book title is "Whispers of the Last Horizon". This book publisher is "Silverbranch Press". This book genres are fantasy, adventure, coming-of-age, young adult, magical realism. The book cover should show a vast twilight landscape with a silhouette of a young girl standing at the edge of a cliff, a glowing fox spirit by her side, and ancient, crumbling towers in the distance under a sky full of swirling stars and auroras, evoking a sense of wonder, mystery, and epic journey.',
    ]'''


    prompt = [#'Book Cover - This book title is "speaker for the dead" . This book publisher is "tor books" . This book Genres tags are science fiction , fiction , fantasy , science fiction fantasy , young adult , audiobook , science fiction , aliens , space , novels , space , space opera . now available in mass market the revised definitive edition of the hugo and nebula classic . in this second book in the saga set years after the terrible war ender wiggin is reviled by history as the xenocide the destroyer of the alien buggers . now ender tells the true story of the war and seeks to stop history from repeating itself . in the aftermath of his terrible war ender wiggin disappeared and a powerful voice arose the speaker for the dead who told the true story of the bugger long years later a second alien race has been discovered but again the aliens ways are strange and frightening again humans die . and it is only the speaker for the dead who is also ender wiggin the xenocide who has the courage to confront the mystery and the for the dead the second novel in orson scott card ender quintet is the winner of the nebula award for best novel and the hugo award for best novel .',
            "Book Cover - This book title is \"love and mistletoe\". This book publisher is \"beaverstone press llc\". This book genres are romance , contemporary , romance , contemporary romance , holiday , christmas , holiday , cultural , ireland , novella , romance , m f romance , business , amazon , new adult.",
            #"Book Cover - This book title is \"stolen songbird\". This book publisher is \"strange chemistry\". This book genres are fantasy , young adult , romance , fantasy , magic , young adult , young adult fantasy , fantasy , paranormal , fantasy , high fantasy , fiction , fairies , fae , paranormal , witches.",
            #"Book Cover - This book title is \"my almost epic summer\". This book publisher is \"putnam juvenile\". This book genres are young adult , young adult , teen , fiction , realistic fiction , audiobook , humor , young adult , coming of age.",
            "Book Cover - This book title is \"aru shah and the tree of wishes\". This book publisher is \"rick riordan presents\". This book genres are childrens , middle grade , fantasy , fantasy , mythology , young adult , fiction , adventure , audiobook , childrens , fantasy , magic , fantasy , urban fantasy.",
            #"Book Cover - This book title is \"a gathering of shadows\". This book publisher is \"tor books\". This book genres are fantasy , young adult , fiction , fantasy , magic , adult , lgbt , audiobook , adventure , romance , fantasy , high fantasy.",
            #"Book Cover - This book title is \"breathe into me\". This book publisher is \"amanda stone\". This book genres are new adult , romance , academic , college , contemporary , romance , contemporary romance , sociology , abuse , young adult , adult , womens fiction , chick lit , realistic fiction.",
            #"Book Cover - This book title is \"surrendered\". This book publisher is \"adler and holt\". This book genres are adult fiction , erotica , romance , romance , contemporary romance , romance , romantic suspense , romance , erotic romance , romantic.",
            #"Book Cover - This book title is \"the white giraffe\". This book publisher is \"dial books young readers\". This book genres are fantasy , animals , fiction , childrens , childrens , middle grade , cultural , africa , adventure , mystery , young adult , childrens , juvenile.",
            'Book Cover - This book title is "speaker for the dead" . This book publisher is "tor books" . This book Genres tags are science fiction , fiction , fantasy , science fiction fantasy , young adult , audiobook , science fiction , aliens , space , novels , space , space opera .',
    ]

    # prompt in later train set
    #prompt = ["Book Cover - This book title is \"aru shah and the tree of wishes\". This book publisher is \"rick riordan presents\". This book genres are childrens , middle grade , fantasy , fantasy , mythology , young adult , fiction , adventure , audiobook , childrens , fantasy , magic , fantasy , urban fantasy. war between the devas and the demons is imminent and the otherworld is on high alert . when intelligence from the human world reveals that the sleeper is holding a powerful clairvoyant and her sister captive aru and her friends launch a mission . the captives a pair of twins turn out to be the newest pandava sisters though according to a prophecy one sister is not the celebration of holi the heavenly attendants stage a massage pr rebranding campaign to convince everyone that the pandavas are to be trusted . as much as aru relishes the attention she fears that she is destined to bring destruction to her sisters as the sleeper has predicted . aru believes that the only way to prove her reputation is to find the kalpavriksha the tree that came out of the ocean of milk when it was churned . if she can reach it before the sleeper perhaps she can turn everything around with one what you wish for aru ."]
    neg_prompt = ['no title "stolen songbird" on the image']

    #prompt = "a pikachu hold the sign that says hello world"





    '''
    input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    print(input_ids.shape)'''

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
    tokenizer_two = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
            model_id,revision= None
        )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
            model_id,revision= None, subfolder="text_encoder_2"
        )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
            model_id, subfolder="text_encoder", revision=None, variant=None
        ).to(device)
    text_encoder_two = text_encoder_cls_two.from_pretrained(
            model_id, subfolder="text_encoder_2", revision=None, variant=None
        ).to(device)
    #print(type(text_encoder_one))
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[tokenizer_one, tokenizer_two],
                        prompt=prompt
                        )
    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[tokenizer_one, tokenizer_two],
                        prompt=neg_prompt, neg_prompt=True, prompt_max_length=prompt_embeds.shape[1]
                        )

    batch = {
        #"original_sizes":[(2048, 1360)],
        "original_sizes":[(350, 425)],
        'crop_top_lefts':[(0, 0)]
    }
    add_time_ids = torch.cat(
    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
    )

    #unet_added_conditions = {"time_ids": add_time_ids}
    unet_added_conditions = {}

    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
    #sys.exit()

    images = pipe(guidance_scale=args.guidance_scale,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                #negative_prompt_embeds=neg_prompt_embeds,
                #negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                height=1152,
                width=896,
                num_inference_steps=50,
                num_images_per_prompt=1).images


    for i, image in enumerate(images):
        image.save(f"{args.image_save_path}/{i}.png")
        print(f"image {str(i)} saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    max_length = 77 # the default max length of CLIP text encoder
    
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--image_save_path', type=str, default='./evaluation_result/images')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--guidance_scale', type=int, default=5)
    args = parser.parse_args()
    device = f"cuda:{str(args.cuda)}"

    main(args)

