# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2-large"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

model.eval()

tokenizer.pad_token = tokenizer.eos_token

dom0 = "Input: premise: ``Oh, my poor Folly... We 've been together for five years, Lexy and I - she practically holds that company together. Of course I gave her an ``A''. But that doesn't mean I'm having an affair with her. [SEP] hypothesis: he is having an affair with Lexy\n Output: contradiction\n"
dom1 = "Input: premise: B: Yeah. Those are pretty. A: Number one turned out just great, and the lady said she couldn't believe that they know that I had done it in the color that they had decorated the nursery [SEP] hypothesis: they know that she had done it in the colors that they had decorated the nursery\n Output: entailment\n"
dom2 = "Input: premise: A: That is the reason, I don't play over there. B: Yeah. A: I like the course, but I don't play over there because, they don't, uh, you know don't allow you to pull a cart. B: Right. A: And, I don't think a cart damages the turf. [SEP] hypothesis: a cart damages the turf\n Output: contradiction\n"
dom3 = "Input: premise: A: or you know, it doesn't seem that it's going to make much of a difference. B: Uh-huh. It, I mean, I don't know, I don't think George Bush will make the American people happy with ninety-seven cents a week. [SEP] hypothesis: George Bush will make American people happy with ninety-seven cents a week\n Output: contradiction\n"

x_q = "Input: premise: A: How did Radio Shack work? B: If you go in and buy anything they want your phone number. And I don't think they're going to call me and ask me how it's functioning, [SEP] hypothesis: they're going to call him\n Output:"

sentence_S = dom1 + dom2 + dom1 + dom0 + dom1 + dom2 + x_q
sentence_S_prime = dom1 + dom2 + dom1 + dom0 + dom1 + dom3 + x_q

# sentence_S = dom1 + dom2
# sentence_S_prime = dom1 + dom3

tokens_S = tokenizer(sentence_S, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
tokens_S_prime = tokenizer(sentence_S_prime, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

with torch.no_grad():
    embedding_S = model.transformer.wte(tokens_S["input_ids"])  # (1, seq_len, hidden_dim)
    embedding_S_prime = model.transformer.wte(tokens_S_prime["input_ids"])  # (1, seq_len, hidden_dim)

embedding_S.requires_grad = True

output_S = model(inputs_embeds=embedding_S).logits  # (1, seq_len, vocab_size)

last_token_idx_S = tokens_S["attention_mask"].sum(dim=1) - 1

if last_token_idx_S + 1 < tokens_S["input_ids"].shape[1]:
    target_token_S = tokens_S["input_ids"][0, last_token_idx_S + 1]
else:
    target_token_S = torch.tensor(tokenizer.eos_token_id).to(output_S.device) 

log_probs_S = torch.nn.functional.log_softmax(output_S[0, last_token_idx_S, :], dim=-1)  # (vocab_size,)

print("log_probs_S shape:", log_probs_S.shape)
print("target_token_S:", target_token_S)

loss_S = -log_probs_S.squeeze(0)[target_token_S]
loss_S.backward()

gradient_LM = embedding_S.grad

delta_P = embedding_S_prime - embedding_S.detach()

taylor_correction = torch.sum(gradient_LM * delta_P).item()
taylor_approx = loss_S.item() + taylor_correction 

with torch.no_grad():
    output_S_prime = model(inputs_embeds=embedding_S_prime).logits
    log_probs_S_prime = torch.nn.functional.log_softmax(output_S_prime[0, last_token_idx_S, :], dim=-1)
    loss_S_prime = -log_probs_S_prime.squeeze(0)[target_token_S].item()  # Fix indexing

error = abs(loss_S_prime - taylor_approx) / loss_S_prime

predicted_token_id = log_probs_S.argmax().item()
predicted_token = tokenizer.decode(predicted_token_id)

print("Predicted next token:", predicted_token)
print("Predicted next token ID:", predicted_token_id)
print("Expected target token ID:", target_token_S.item())

print("Actual LM(P(S', xq)):", loss_S_prime)
print("Taylor Approximation:", taylor_approx)
print("⟨∇LM(P(S, xq)), P(S', xq) - P(S, xq)⟩:", taylor_correction)
print("Taylor Expansion Error:", error)
