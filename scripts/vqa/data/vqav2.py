""" Dataset object for VQAv2 dataset. """

import json
import re

import h5py
import torch
from torch.utils.data import Dataset


SPECIAL_CHARS = re.compile("[^a-z0-9 ]*")
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCTUATION_CHARS = re.escape(r';/[]"{}()=+\_-><@`,?!')
PUNCTUATION = re.compile(r"([{}])".format(re.escape(PUNCTUATION_CHARS)))
PUNCTUATION_WITH_A_SPACE = re.compile(
    r"(?<= )([{0}])|([{0}])(?= )".format(PUNCTUATION_CHARS)
)


class VQA(Dataset):
    """ VQAv2 dataset. """

    def __init__(
        self,
        questions_path: str,
        answers_path: str,
        image_features_path: str,
        vocab_path: str,
        answerable_only: bool = False,
        dummy_answers: bool = False,
    ):
        """ Init function for VQA. """

        super(VQA, self).__init__()

        # Read in questions, answers, and vocab.
        with open(questions_path, "r") as fd:
            questions_json = json.load(fd)
        with open(answers_path, "r") as fd:
            answers_json = json.load(fd)
        with open(vocabulary_path, "r") as fd:
            vocab_json = json.load(fd)

        self.question_ids = [q["question_id"] for q in questions_json["questions"]]

        # Process questions and answers.
        self.questions = list(prepare_questions(questions_json))
        self.answers = list(prepare_answers(answers_json))
        self.questions = [self._encode_question(q) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers]

        # Process image features.
        self.image_features_path = image_features_path
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q["image_id"] for q in questions_json["questions"]]

        # Process vocab.
        self.vocab = vocab_json
        self.token_to_index = self.vocab["question"]
        self.answer_to_index = self.vocab["answer"]

        # Handle using answerable questions only or using dummy answers.
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable(not self.answerable_only)
        self.dummy_answers = dummy_answers

    @property
    def max_question_length(self):
        if not hasattr(self, "_max_length"):
            self._max_length = max(map(len, self.questions))
        return self._max_length

    @property
    def num_tokens(self):
        # The extra 1 here is for the <unknown> token at index 0.
        return len(self.token_to_index) + 1

    def _create_coco_id_to_index(self):
        """
        Returns a mapping from a COCO image id into the corresponding index into the h5
        file.
        """
        with h5py.File(self.image_features_path, "r") as features_file:
            coco_ids = features_file["ids"][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _check_integrity(self, questions, answers):
        """ Verify that questions and answers are consistent. """
        qa_pairs = list(zip(questions["questions"], answers["annotations"]))
        assert all(
            q["question_id"] == a["question_id"] for q, a in qa_pairs
        ), "Questions not aligned with answers"
        assert all(
            q["image_id"] == a["image_id"] for q, a in qa_pairs
        ), "Image id of question and answer don't match"
        assert questions["data_type"] == answers["data_type"], "Mismatched data types"
        assert (
            questions["data_subtype"] == answers["data_subtype"]
        ), "Mismatched data subtypes"

    def _find_answerable(self, count=False):
        """
        Returns a list of indices into questions that have at least one answer in the
        vocab.
        """
        answerable = []
        if count:
            number_indices = torch.LongTensor(
                [self.answer_to_index[str(i)] for i in range(0, 8)]
            )
        for i, answers in enumerate(self.answers):
            # store the indices of anything that is answerable
            if count:
                answers = answers[number_indices]
            answer_has_index = len(answers.nonzero()) > 0
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _encode_question(self, question):
        """ Encode a question as a vector of indices and a question length. """
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def _encode_answers(self, answers):
        """
        Encode an answer as a vector. The result will be a vector of answer counts to
        determine which answers will contribute to the loss. This should be multiplied
        with 0.1 * negative log-likelihoods that a model produces and then summed up to
        get the loss that is weighted by how many humans gave that answer.
        """
        answer_vec = torch.zeros(len(self.answer_to_index))
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def _load_image(self, image_id):
        """ Load an image. """

        # Loading the h5 file has to be done here and not in __init__ because when the
        # DataLoader forks for multiple workers, every child would use the same file
        # object and fail. Having multiple readers using different file objects is fine
        # though, so we just init in here.
        if not hasattr(self, "features_file"):
            self.features_file = h5py.File(self.image_features_path, "r")
        index = self.coco_id_to_index[image_id]
        img = self.features_file["features"][index]
        boxes = self.features_file["boxes"][index]
        return torch.from_numpy(img).unsqueeze(1), torch.from_numpy(boxes)

    def __getitem__(self, item):
        """ Return the dataset element with index `item`. """

        # Handle case of using answerable questions only.
        if self.answerable_only:
            item = self.answerable[item]

        # Get question, answer, and visual features + bounding boxes.
        q, q_length = self.questions[item]
        a = 0 if self.dummy_answers else self.answers[item]
        image_id = self.coco_ids[item]
        v, b = self._load_image(image_id)

        # Since batches are re-ordered for PackedSequence's, the original question order
        # is lost. We return `item` so that the order of (v, q, a) triples can be
        # restored if desired. Without shuffling in the dataloader, these will be in the
        # order that they appear in the q and a json files.
        return v, q, a, b, item, q_length

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [q["question"] for q in questions_json["questions"]]
    for question in questions:
        question = question.lower()[:-1]
        question = SPECIAL_CHARS.sub("", question)
        yield question.split(" ")


def prepare_answers(answers_json):
    """
    Normalize answers from a given answer json in the usual VQA format. The only
    normalization that is applied to both machine generated answers as well as ground
    truth answers is replacing most punctuation with space (see [0] and [1]). Since
    potential machine generated answers are just taken from most common answers,
    applying the other normalizations is not needed, assuming that the human answers are
    already normalized.

    [0]: http://visualqa.org/evaluation.html
    [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96
    """
    answers = [
        [a["answer"] for a in ans_dict["answers"]]
        for ans_dict in answers_json["annotations"]
    ]

    # The original is somewhat broken, so things that look odd here might just be to
    # mimic that behaviour. This version should be faster since we use re instead of
    # repeated operations on strings.
    def process_punctuation(s):
        if PUNCTUATION.search(s) is None:
            return s
        s = PUNCTUATION_WITH_A_SPACE.sub("", s)
        if re.search(COMMA_STRIP, s) is not None:
            s = s.replace(",", "")
        s = PUNCTUATION.sub(" ", s)
        s = PERIOD_STRIP.sub("", s)
        return s.strip()

    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))
