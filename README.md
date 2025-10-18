
# LLM Search Code
The `langagent` package implements modular implementations of popular LLM search algorithms, e.g., Tree-Of-Thoughts, Reasoning via Planning (RAP).


## Usage
See `examples/math_qa/main_search.py` for an example of using the `langagent` package.
The entry point is the `main` function.
```python
main(
    dataset_name="math500", 
    model_name=model_name,
    eval_model_name=eval_model_name, 
    reasoning_method="bfs", 
    add_continuation=False, 
    bn_method=None, 
    bn_model_name=bn_model_name, 
    eval_idx=eval_idx
)
```
* dataset_name: "math500", "gsm8k"
* model_name: name of the model to be used for search. Since we use Huggingface Transformers, you may need to set you own HF token. Please locate the following code snippet the `main_search.py` file to add you token.
    ```
    Add you own Hf token
    from huggingface_hub import login
    hf_token = ""
    login(token = '')
    ```
* eval_model_name: name of the model to be used for evaluation
* reasoning_method: "bfs", "rap", "rest"
* add_continuation: whether to add chaining to the search process
* bn_method: "direct", "entropy" (corresponding to sc1 in the paper), "sc" (corresponding to sc2 in the paper)
* bn_model_name: name of the model to be used for continuation
* eval_idx: list of indices of examples to be evaluated. By default, all the 100 examples are evaluated.
