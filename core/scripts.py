import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from loguru import logger
from rouge_score import rouge_scorer
import g4f
from g4f.client import Client

logger.add("out.log", diagnose=False)


class GPT2Model:

    def __init__(self, model_name_or_path: str):

        """
            Initialize the GPT2Summarizer class.

            Args:
                model_name_or_path (str): Name or path of the pre-trained GPT-2 model.
        """
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.score = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.score_main_params = 'rouge1'

        self.config_tokenizer = {
            'max_length': 600,
            'add_special_tokens': False,
            'padding': False,
            'truncation': True
        }

    def generate_by_conf_variants(self, text: str) -> torch.Tensor:

        """
            Generate summaries based on different configurations.

            Args:
                text (str): Input text.

            Returns:
                torch.Tensor: Output token IDs.
        """
        text_tokens = self.tokenizer(
            text,
            **self.config_tokenizer
        )["input_ids"]

        input_ids = text_tokens + [self.tokenizer.sep_token_id]
        input_ids = torch.LongTensor([input_ids])

        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=7,
            num_return_sequences=3,
        )

        return output_ids

    def decoding_output_ids(self, output_ids: torch.Tensor) -> list:
        """
           Decode output token IDs to summaries.

           Args:
               output_ids (torch.Tensor): Output token IDs.

           Returns:
               list: Decoded summaries.
        """
        summaries = []
        for id in output_ids:
            summary = self.tokenizer.decode(id, skip_special_tokens=False)
            summary = summary.split(self.tokenizer.sep_token)[1]
            summary = summary.split(self.tokenizer.eos_token)[0]
            summaries.append(summary)
        return summaries

    def calc_score(self, text, summaries):
        scores = []
        for index, summary in enumerate(summaries):
            scores.append({
                'index': index,
                'text': summary,
                'score': self.score.score(text, summary)
            })
        return scores

    def __transform_scores(self, scores: list) -> dict:
        """
            Transform scores into a more accessible format.

            Args:
                scores (list): List of dictionaries containing scores.

            Returns:
                dict: Transformed scores.
        """

        return {
            _score['index']: {k: v._asdict() for k, v in _score['score'].items()}
            for _score in scores
        }

    def choose_summary(self, scores: list) -> str:
        """
            Choose the best summary based on Rouge scores.

            Args:
                scores (list): List of dictionaries containing scores.

            Returns:
                str: Best summary.
        """
        result = scores[0]['text']
        transform_scores = self.__transform_scores(scores)
        logger.info(transform_scores)
        max_value = 0
        max_index = None

        for index, x in transform_scores.items():
            if max_value < x[self.score_main_params]['precision']:
                max_value = x[self.score_main_params]['precision']
                max_index = index

        logger.info(f'{max_value} {max_index}')

        if (max_index != None and max_value != 0):
            result = scores[max_index]['text']

        return result

    def generate_summary_by_text(self, text: str) -> str:
        """
            Generate a summary for a given text.

            Args:
                text (str): Input text.

            Returns:
                str: Generated summary.
        """

        output_ids = self.generate_by_conf_variants(text)
        summarys = self.decoding_output_ids(output_ids)
        scores = self.calc_score(text, summarys)
        result = self.choose_summary(scores)

        logger.debug(f'summarys -> {summarys}')
        logger.debug(f'scores -> {scores}')

        return result


class GPT:

    PROMPT_TEMPLATE = ("""Сделай лид новости, состоящий не более чем из 10 слов на русском языке, для следующей новости: {}. "
                       "Не добавляй никаких знаков, эмоджи, вступлений. Выведи лид одним предложеним. Используй необходимые знаки препинания. "
                       "Не добавляй в начале вывода что-то наподобие "Возможный лид новости:" или "Лид новости:". Не используй Copilot. Не добавляй ничего к лиду""")

    def __init__(self):
        self.gpt3 = "gpt-3.5-turbo"
        self.gpt4 = "gpt-4"

    async def get_gpt3(self, text: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(text)
        response = await g4f.ChatCompletion.create_async(
            model=self.gpt3,
            messages=[{"role": "user", "content": prompt}])
        return response

    async def get_gpt4(self, text: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(text)
        response = await g4f.ChatCompletion.create_async(
            model=self.gpt4,
            messages=[{"role": "user", "content": prompt}])
        return response