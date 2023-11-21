import openai
import replicate
import json
from tqdm import tqdm
# from openai import OpenAI
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import time
import os
import numpy as np
import random
import argparse

# openai.api_key = "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"
openai.api_base = "https://api.lmtchina.com/v1"
openai.api_key = "sk-LdjWHcUY33JHzwsU43F970A5705742098e22F9B73c366f56"


def get_response_with_openai_api(model_name, inputs, shots):
    # import openai
    # openai.api_key = "EMPTY"  # Not support yet
    # openai.api_base = "http://localhost:8001/v1"
    # openai.api_version = "2023-07-01-preview"
    # openai.api_key = key
    
    openai.api_base = "https://api.lmtchina.com/v1"
    openai.api_key = "sk-LdjWHcUY33JHzwsU43F970A5705742098e22F9B73c366f56"
    # print(inputs["gold_answer"])
    if shots == 4:
        prompt = inputs.get('base_words') + inputs.get('examples') + inputs.get('prompt')
    else:
        prompt = inputs.get('prompt')
    # print(prompt)
    # print("*************")
    # # history = inputs.get('history')
    # max_length = inputs.get('max_length')
    # top_p = inputs.get('top_p')
    # temperature = inputs.get('temperature')
    temperature = 0.75
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            temperature=temperature if temperature else 0,
            messages=[
                {"role": "user", "content": prompt}]
        )
    except json.decoder.JSONDecodeError:
        time.sleep(5)
        response = openai.ChatCompletion.create(
            model=model_name,
            temperature=temperature if temperature else 0,
            messages=[
                {"role": "user", "content": prompt}]
        )
    predict = response.choices[0].message.content
    inputs["predict"] = predict
    print(f"\n Prediction:\2 {predict} \2 Golden:{inputs.get('gold_answer')}")
    print("\n")

    return inputs


def LlaMa_test(inputs, shot, use):
    if shot == 0:
        output = replicate.run(
            "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
            input={"prompt": inputs.get("dialogue"), "system_prompt": inputs.get("system_prompt"), "temperature": 0.75 ,
                   "seed": 4321}
        )
    else:
        if use:
            output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": inputs.get("dialogue"), "system_prompt": inputs.get("base_words") +
                                                                      inputs.get("llama_examples") +
                                                                      inputs.get("system_prompt"), "temperature": 0.75,
                   "seed": 12345}
        )
        else:
            output = replicate.run(
                "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
                input={"prompt": inputs.get("dialogue"), "system_prompt": inputs.get("base_words") +
                                                                        inputs.get("llama_examples") +
                                                                        inputs.get("system_prompt"), "temperature": 0.75,
                    "seed": 1234}
            )
    # The meta/llama-2-13b-chat model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    # print("Prediction")
    results = []
    predict = ''
    for item in output:
        predict += item
    
    inputs["predict"] = predict
    print(f"\n Prediction:\2 {predict} \2 Golden:{inputs.get('gold_answer')}")
    print("\n")
    return inputs


def input_formation(path, model_name, shot):
 
    with open(path, 'r') as f:
        data = json.load(f)
    # print(len(data))
    inputs = []
    input = {}
    shot_examlpe = {}
    for single in data:
        input['gold_answer'] = single.get('answer')
        input['base_words'] = "You are a customer service consultant. Give a proper response to answer the " \
                                 "initial question proposed by the client based on the given document. " \
                                 "You can ONLY reply ’Irrelevant’, ’Yes’, ’No’ directly or ask a follow-up question " \
                                 "without explanation. Don’t explain! Don’t explain! Don’t explain! \n\n "
        
        # for llama
        input['system_prompt'] = "You are a customer service consultant. Give a proper response to answer the " \
                                 "initial question proposed by the client based on the given document. " \
                                 "You can ONLY reply ’Irrelevant’, ’Yes’, ’No’ directly or ask a follow-up question " \
                                 "without explanation. Don’t explain! Don’t explain! Don’t explain! \n\n " + f"Document: \n {single.get('snippet')} \n\n "
        
        # gpt examples
        shot_file = 'examples/' + str(shot) + 'shot.json'
        with open(shot_file, 'r') as f:
            data = json.load(f)       
        input['examples'] = '' 
        for example in data:
            input['examples'] += example
        
        '''
        input['llama_examples'] = "### Example \n\n Document: Eligibility \n\n You must have: " \
                        "\n• an unconditional offer of a place on a course with a licensed Tier 4 sponsor" \
                        "\n• enough money to support yourself and pay for your course - the amount will vary " \
                        "depending on your circumstances." \
                        "\n\n[INST]I have an unconditional offer for a course with a licensed tier " \
                        "with a licensed Tier 4 sponsor who is not a panel of experts." \
                        " Am I eligible for a Tier 4 (General) student visa?[/INST]" \
                        "\nDo you have an unconditional offer of a place on a course"\
                        "\n[INST]Yes.[/INST]]" \
                        "\nDo you have enough money to support yourself and pay for the course?" \
                        "\n[INST]No.[/INST]" \
                        "\n\nThe final response: No. \n\n\n" \
                        "### Example \n\nDocument: " \
                        "\n In order to qualify for this benefit program, your business or private " \
                        "non-profit organization must have sustained " \
                        "physical damage and be located in " \
                        "a disaster declared county." \
                        "\n\n[INST]My housing benefit doesn't currently cover my rent. " \
                        "Does this program meet my needs?[/INST]" \
                        "\n\nThe final response: Do you own a business or private " \
                        "non-profit organization? \n\n\n" \
                        "### Example \n\nDocument: \n\n" \
                        "## Taking more leave than the entitlement\n\n" \
                        "If a worker has taken more leave than they're entitled to, " \
                        "their employer must not take money from their final pay unless it's been agreed " \
                        "beforehand in writing. The rules in this situation should be outlined in the " \
                        "employment contract, company handbook or intranet site." \
                        "\n\n[INST]Have I logged out properly?[/INST]" \
                        "\n\nThe final response: Irrelevant. \n\n\n" \
                        "### Example \n\nDocument: \n\n" \
                        "## Items that qualify for the zero rate\n" \
                        "The eligible items include:\n* medical, veterinary and scientific equipment" \
                        "\n* ambulances\n* goods for disabled people\n* motor vehicles for medical use" \
                        "\n\n[INST]Is this item eligible?[/INST]" \
                        "\n\nThe final response: Yes."
        
        shot_examlpe['examples'] = input['examples']
        examples = []
        examples.append(shot_examlpe['examples'])
        with open('last_4shot.json', 'w') as ff:
            json.dump(examples, ff)
        '''
        prompt = f"You are a customer service consultant. Give a proper response to answer the initial question " \
                 f"proposed by the client based on the given document. You can ONLY reply ’Irrelevant’, ’Yes’, ’No’ " \
                 f"directly or ask a follow-up question without explanation. Don’t explain! Don’t explain! " \
                 f"Don’t explain! \n\n " \
                 f"Document: \n {single.get('snippet')} \n\n" \
                 f"Initial question: {single.get('scenario')} {single.get('question')} \n\n "
        
        # add initial question for llama
        input['dialogue'] = f"[INST]{single.get('scenario')} {single.get('question')}[/INST] \n"
        # create dialogue
        follow_ups = len(single['history'])
        # print(follow_ups)
        if follow_ups == 0:
            prompt += "No conversation has taken place."
            input['max_token'] = 100
            input['temperature'] = 0.75
        else:
            prompt += "The following is the conversation that has already happened: \n\n"
            for i in range(follow_ups):
                if "gpt" in model_name:
                    try:
                        prompt += f"You: {single['history'][i]['follow_up_question']} \n" \
                                  f"Client:{single['history'][i]['follow_up_answer']} \n"
                    except IndexError:
                        print(i, single['history'])

                elif model_name == "llama":
                    try:
                        input['dialogue'] += f"{single['history'][i]['follow_up_question']} \n " \
                                             f"[INST]{single['history'][i]['follow_up_answer']}[/INST] \n "
                    except IndexError:
                        print(i, single['history'])

            input['temperature'] = 0.75
            input['max_token'] = 10
        input['top_p'] = 0.75
        input['prompt'] = prompt
        inputs.append(input)
        input = {}
    # print(inputs[0])
    return inputs

def add_example(shot):
    train_file = 'sharc1-official/json/sharc_train.json'
    yes_answer, no_answer, irrelevant_answer, ask_answer = [], [], [], []
    with open(train_file, 'r') as ff:
        data = json.load(ff)
    for single in data:
        if 'Yes' in single['answer']:
            yes_answer.append(single)
        elif 'No' in single['answer']:
            no_answer.append(single)
        elif 'Irrelevant' in single['answer']:
            irrelevant_answer.append(single)
        else:
            ask_answer.append(single)
    
    per_example = int((shot-4)/4)
    shots, cache = [], [[], [], [], []]
    for i in range(per_example):
        while 1:
            rand_yes = random.randint(0, len(yes_answer)-1)
            if rand_yes not in cache[0]:
                break
        shots.append(yes_answer[rand_yes])
        cache[0].append(rand_yes)
        
        while 1:
            rand_no = random.randint(0, len(no_answer)-1)
            if rand_no not in cache[1]:
                break
        shots.append(no_answer[rand_no])
        cache[1].append(rand_no)
        
        while 1:
            rand_irr = random.randint(0, len(irrelevant_answer)-1)
            if rand_irr not in cache[2]:
                break
        shots.append(irrelevant_answer[rand_irr])
        cache[2].append(rand_irr)
        
        while 1:
            rand_ask = random.randint(0, len(ask_answer)-1)
            if rand_ask not in cache[3]:
                break
        shots.append(ask_answer[rand_ask])
        cache[3].append(rand_ask)

    examples = []
    for example_data in shots:
        example = f"### Example Document:{example_data.get('snippet')} \n\nInitial question:{example_data.get('scenario')}{example_data.get('question')}\n\n"
        follow_ups = len(example_data.get('history'))
        # print(follow_ups)
        if follow_ups == 0:
            example += "No conversation has taken place. \n\n"
        else:
            example += "The following is the conversation that has already happened: \n"
            for i in range(follow_ups):
                try:
                    example += f"You: {example_data['history'][i]['follow_up_question']} \n" \
                                f"Client:{example_data['history'][i]['follow_up_answer']} \n"
                except IndexError:
                    print(i, single['history'])  
        example += f"\nThe final response:{example_data.get('answer')}\n\n\n"
        examples.append(example)
    
    with open('examples/4shot.json', 'r') as ff:
        shot_4 = json.load(ff)
    examples.extend(shot_4)

    save_file = 'examples/' + str(shot) + 'shot.json'
    with open(save_file, 'w') as ff:
        json.dump(examples, ff)
    

def evaluate(data):
    # with open(file, 'r') as fr:
    #     data = json.load(fr)
    whole_correct, yes_correct, no_correct, irr_correct, ask_correct = 0, 0, 0, 0, 0
    # wrong_yes, wrong_no, wrong_irrelevant, wrong_ask = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    correct, predict = [], []
    asks = []
    counts = 0
    generate = []
    gold_span = []
    for single in data:
        counts += 1
        gold = single['gold_answer']
        prediction = single['predict']

        if 'Yes' in gold:
            correct.append(1)
        elif 'No' in gold:
            correct.append(2)
        elif 'Irrelevant' in gold:
            # print(gold)
            correct.append(3)
        else:
            correct.append(4)

        if ('No' not in prediction) and ('not' not in prediction) and ('Irrelevant' not in prediction) \
                and ('?' not in prediction):
            predict.append(1)
        elif ('No' in prediction) and ('?' not in prediction):
            predict.append(2)
        elif 'Irrelevant' in prediction and ('?' not in prediction):
            # print(prediction)
            predict.append(3)
        else:
            predict.append(4)
            gold_span.append([gold.split()])
            generate.append(prediction.split())


        # if '?' in gold and '?' in prediction:
        #     # print(gold, prediction)
        #     gold_span.append([gold.split()])
        #     generate.append(prediction.split())

        prediction = single['predict']
    # print(correct, predict)
    lable = [1, 2, 3, 4]
    f1_micro = f1_score(correct, predict, labels=lable, average="micro")
    f1_macro = f1_score(correct, predict, labels=lable, average="macro")
    f1_weighted = f1_score(correct, predict, labels=lable, average="weighted")
    print(f1_micro, f1_macro, f1_weighted)
    conf = confusion_matrix(correct, predict)
    print(conf)
    print(counts, whole_correct, yes_correct, no_correct, irr_correct, ask_correct)
    # print(wrong_yes, wrong_no, wrong_irrelevant, wrong_ask)
    
    # BLEU calculate
    sentence_bleu_1, sentence_bleu_4 = [], []
    for i in range(len(gold_span)):
        sentence_bleu_1.append(sentence_bleu(gold_span[i], generate[i], weights=(1, 0, 0, 0)))
        sentence_bleu_4.append(sentence_bleu(gold_span[i], generate[i], weights=(0.25, 0.25, 0.25, 0.25)))
    print(len(gold_span))
    print("Avg BLEU-1 Score:", sum(sentence_bleu_1)/len(sentence_bleu_1))
    print("Avg BLEU-4 Score:", sum(sentence_bleu_4)/len(sentence_bleu_4))

    corpus_bleu_1 = corpus_bleu(gold_span, generate, weights=(1, 0, 0, 0))
    corpus_bleu_4 = corpus_bleu(gold_span, generate, weights=(0.25, 0.25, 0.25, 0.25))

    print("Corpus BLEU-1 Score:", corpus_bleu_1)
    print("Corpus BLEU-4 Score:", corpus_bleu_4)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--shot', type=int, default=False, required=True, help='shot number, restricted in 0, 4, 8, 16, 32, 64, 128, 256')
    parser.add_argument('--run', type=bool, default=False, required=False, help='run')
    parser.add_argument('--evaluate', type=bool, required=False, help='evaluate')
    args = parser.parse_args()

    train_file = 'sharc1-official/json/sharc_train.json'
    data_path = 'sharc1-official/json/sharc_dev.json'
    model_class = ['gpt-3.5-turbo', 'llama', 'gpt-4-1106-preview']
    shots = [0, 4, 8, 16, 32, 64, 128, 256]
    
    # prepare examples

    # inputs = input_formation(data_path, model_name=model_class[2], shot=shots[1])
    # for i in range(2, len(shots)):
    #     print(i, shots[i])
    #     add_example(shots[i])
    
    run = args.run
    # form input
    shot = args.shot
    model_name = model_class[2]
    
    if run:
        inputs = input_formation(data_path, model_name=model_name, shot=shot)
        outputs = []
        data_len = 0
        if "gpt" in model_name:
            # find stop point
            count = 0
            for i in range(10):
                file_name = model_name + "/" + model_name + "_" + str(shot) + "shot_" + str(i) + ".json"
                if os.path.exists(file_name):
                    count += 1
                    with open(file_name, 'r') as ff:
                        data = json.load(ff)
                    data_len += len(data)
            
            i = 0
            for input in tqdm(inputs):              
                if i < data_len:
                    i += 1
                    continue
                output = get_response_with_openai_api(model_name, input, shot)
                # print("output", output)
                outputs.append(output)
                save_file = model_name + "/" + model_name + "_" + str(shot) + "shot_" + str(count) +".json"
                with open(save_file, "w") as f1:
                    json.dump(outputs, f1)
                f1.close()
        else:
            use_70 = True

            # if use_70:
            #     save_file = 'full_outputs_4shots_70b.json'
            #     with open(save_file, "r") as f2:
            #         data = json.load(f2)
            #     data_len = len(data)
            # else:
            #     save_file = 'full_outputs_4shots.json'
            #     with open(save_file, "r") as f1:
            #         data = json.load(f1)
            #     retend_file = 'full_outputs_4shots_retend.json'
            #     with open(retend_file, 'r') as f2:
            #         retend_data = json.load(f2)
            #     data_len = len(data) + len(retend_data)

            # i = 0
            # for input in tqdm(inputs):
            #     # print("input:", input)
            #     # get_response_with_openai_api(model_name, input)
            #     if i < data_len:
            #         i += 1
            #         continue
            #     output = LlaMa_test(input, shots[1], use_70)
            #     # print("output", output)
            #     outputs.append(output)
            #     if use_70:
            #         save_file = 'full_outputs_4shots_70b_retend.json'
            #         with open(save_file, "w") as f1:
            #             json.dump(outputs, f1)
            #         f1.close()
            #     else:
            #         save_file = 'full_outputs_4shots_retennd_2.json'
            #         with open(save_file, "w") as f1:
            #             json.dump(outputs, f1)
            #         f1.close()
            # print(len(outputs), outputs[0])
            # if use_70:
            #     save_file = 'full_outputs_4shots_70b_full.json'
            #     with open(save_file, "w") as f1:
            #         json.dump(outputs, f1)
            #     f1.close()
            # else:
            #     save_file = 'full_outputs_4shots_full.json'
            #     with open(save_file, "w") as f1:
            #         json.dump(outputs, f1)
            #     f1.close()
    
    evaluation = args.evaluate
    if evaluation:
        if model_name == model_class[1]:
            use_70 = False
            if use_70:
                file_1 = 'full_outputs_4shots_70b.json'
                file_2 = 'full_outputs_4shots_70b_retend.json'
                with open(file_1, 'r') as ff1:
                    data_1 = json.load(ff1)
                with open(file_2, 'r') as ff2:
                    data_2 = json.load(ff2)
                full_data = []
                full_data.extend(data_1)
                full_data.extend(data_2)
                print(len(full_data))
                evaluate(full_data)
            else:
                file_1 = 'full_outputs_4shots.json'
                file_2 = 'full_outputs_4shots_retend.json'
                file_3 = 'full_outputs_4shots_retennd_2.json'
                with open(file_1, 'r') as ff1:
                    data_1 = json.load(ff1)
                with open(file_2, 'r') as ff2:
                    data_2 = json.load(ff2)
                with open(file_3, 'r') as ff3:
                    data_3 = json.load(ff3)
                full_data = []
                full_data.extend(data_1)
                full_data.extend(data_2)
                full_data.extend(data_3)
                print(len(full_data))
                evaluate(full_data)
        else:
            full_data = []            
            for i in range(10):             
                file_name = model_name + "/" + model_name + "_" + str(shot) + "shot_" + str(i) +".json"
                if os.path.exists(file_name):
                    with open(file_name, 'r') as ff:
                        data = json.load(ff)
                    full_data.extend(data)
                else:
                    continue

            print(len(full_data))
            print(full_data[0], type(full_data))
            evaluate(full_data)

    
