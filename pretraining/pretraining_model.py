import torch
from torch import nn
from transformers import PreTrainedModel
import os
from transformers import BertModel, RobertaModel, LongformerModel

class EntityAwarePreTrainingModel(PreTrainedModel):
    def __init__(self, config, backbone_model):
        super().__init__(config)
        self.config = config
        self.backbone_model = backbone_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlm = nn.Linear(config.hidden_size, config.vocab_size)
        self.entity_type_predictions = nn.Linear(config.hidden_size, config.num_entity_types)
        self.task_weights = nn.Parameter(torch.ones(2))
        # Initialize weights
        self.init_weights()
        
    def save_pretrained(self, save_directory):
        """ Save model to the specified directory """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the config
        self.config.save_pretrained(save_directory)


        model_to_save = self.module if hasattr(self, 'module') else self
        state_dict = model_to_save.state_dict()


        if isinstance(self.backbone_model, BertModel):

            state_dict = {key.replace("backbone_model.", "bert."): value for key, value in state_dict.items()}
        elif isinstance(self.backbone_model, RobertaModel):

            state_dict = {key.replace("backbone_model.", "roberta."): value for key, value in state_dict.items()}
        elif isinstance(self.backbone_model, LongformerModel):

            state_dict = {key.replace("backbone_model.", "longformer."): value for key, value in state_dict.items()}
        else:
            raise ValueError(f"Unsupported backbone model type: {type(self.backbone_model)}")


        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, config=None, backbone_model=None, *inputs, **kwargs):
        """Load model from pretrained directory"""
        if not config:
            config = cls.config_class.from_pretrained(pretrained_model_path)


        model = cls(config, backbone_model)


        state_dict_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")


            if isinstance(model.backbone_model, BertModel):

                state_dict = {key.replace("bert.", "backbone_model."): value for key, value in state_dict.items()}
            elif isinstance(model.backbone_model, RobertaModel):

                state_dict = {key.replace("roberta.", "backbone_model."): value for key, value in state_dict.items()}
            elif isinstance(model.backbone_model, LongformerModel):

                state_dict = {key.replace("longformer.", "backbone_model."): value for key, value in state_dict.items()}
            else:
                raise ValueError(f"Unsupported backbone model type: {type(model.backbone_model)}")


            model.load_state_dict(state_dict)

        return model



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mlm_labels=None,
            entity_type_labels=None,
    ):
        outputs = self.backbone_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)  # 应用 Dropout 防止过拟合
        
        total_loss = []
        loss_dict = {}


        prediction_scores = self.mlm(sequence_output)  # (batch_size, sequence_length, vocab_size)
        mlm_loss = self.calc_masked_lm_loss(prediction_scores, mlm_labels)
        total_loss.append(mlm_loss)
        loss_dict["mlm_loss"] = mlm_loss.detach().cpu().item()


        entity_type_scores = self.entity_type_predictions(sequence_output)  # (batch_size, sequence_length, num_entity_types)
        entity_type_loss = self.calc_multilabel_loss(entity_type_scores, entity_type_labels)
        total_loss.append(entity_type_loss)
        loss_dict["entity_type_loss"] = entity_type_loss.detach().cpu().item()

        
        total_loss = self.combine_losses(total_loss)
        loss_dict["total_loss"] = total_loss.detach().cpu().item()
        loss_dict["task_weights"] = self.task_weights.detach().cpu().tolist()

        return {
            "total_loss": total_loss,
            "mlm_loss": mlm_loss,
            "entity_type_loss": entity_type_loss,
            "logits": entity_type_scores,
            "task_weights": self.task_weights.detach().cpu().tolist()
        }

    def calc_masked_lm_loss(self, prediction_scores, labels):
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    def calc_multilabel_loss(self, scores, labels):
        loss_fct = nn.BCEWithLogitsLoss()
        active_loss = labels.sum(dim=-1) != 0  # 仅在存在标签的位置计算损失
        if active_loss.any():
            active_scores = scores[active_loss]
            active_labels = labels[active_loss]
            loss = loss_fct(active_scores, active_labels)
            return loss
        else:
            return torch.tensor(0.0, device=scores.device)

    def combine_losses(self, losses):
        weighted_losses = [w * l for w, l in zip(self.task_weights, losses) if w > 0]
        return sum(weighted_losses) / sum(self.task_weights[self.task_weights > 0])