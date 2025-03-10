# Product Classification

In this test we ask you to build a model to classify products into their categories according to their features.

## Dataset Description

The dataset is a simplified version of [Amazon 2018](https://jmcauley.ucsd.edu/data/amazon/), only containing products and their descriptions.

The dataset consists of a jsonl file where each is a json string describing a product.

Example of a product in the dataset:
```json
{
 "also_buy": ["B071WSK6R8", "B006K8N5WQ", "B01ASDJLX0", "B00658TPYI"],
 "also_view": [],
 "asin": "B00N31IGPO",
 "brand": "Speed Dealer Customs",
 "category": ["Automotive", "Replacement Parts", "Shocks, Struts & Suspension", "Tie Rod Ends & Parts", "Tie Rod Ends"],
 "description": ["Universal heim joint tie rod weld in tube adapter bung. Made in the USA by Speed Dealer Customs. Tube adapter measurements are as in the title, please contact us about any questions you 
may have."],
 "feature": ["Completely CNC machined 1045 Steel", "Single RH Tube Adapter", "Thread: 3/4-16", "O.D.: 1-1/4", "Fits 1-1/4\" tube with .120\" wall thickness"],
 "image": [],
 "price": "",
 "title": "3/4-16 RH Weld In Threaded Heim Joint Tube Adapter Bung for 1-1/4&quot; Dia by .120 Wall Tube",
 "main_cat": "Automotive"
}
```

### Field description
- also_buy/also_view: IDs of related products
- asin: ID of the product
- brand: brand of the product
- category: list of categories the product belong to, usually in hierarchical order
- description: description of the product
- feature: bullet point format features of the product
- image: url of product images (migth be empty)
- price: price in US dollars (might be empty)
- title: name of the product
- main_cat: main category of the product

`main_cat` can have one of the following values:
```json
["All Electronics",
 "Amazon Fashion",
 "Amazon Home",
 "Arts, Crafts & Sewing",
 "Automotive",
 "Books",
 "Buy a Kindle",
 "Camera & Photo",
 "Cell Phones & Accessories",
 "Computers",
 "Digital Music",
 "Grocery",
 "Health & Personal Care",
 "Home Audio & Theater",
 "Industrial & Scientific",
 "Movies & TV",
 "Musical Instruments",
 "Office Products",
 "Pet Supplies",
 "Sports & Outdoors",
 "Tools & Home Improvement",
 "Toys & Games",
 "Video Games"]
```

[Download dataset](https://drive.google.com/file/d/1Zf0Kdby-FHLdNXatMP0AD2nY0h-cjas3/view?usp=sharing)

Data can be read directly from the gzip file as:
```python
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
```

## Description of the solution

To deal with this task, two models have been trained:

* TF-IDF Vectorizer + Naive Bayes algorithm. This should have been a simple benchmark to test more complex models on, as it requires little computational power. Its performance has been surprisingly high, though (almost 75% of accuracy/precision and recall on the validation dataset). Note that Naive Bayes, which is a cheap and simple algorithm, relies 
upon the assumption of conditional independence among features (in this case, words). This might not be the case as words are semantically related to each other. However, as sample size increases, this assumption seems more reasonable in a large dimensional space.

* Large Language Model (tiny BERT, https://huggingface.co/prajjwal1/bert-tiny) fine-tuned using this dataset. This was a sensible choice to train with using a Apple M3 chip with 8GB RAM. Memory limitations imposed rather large training times with small batch sizes. Overfitting is an issue as only a fraction of the sample size could be reasonably used. With increasing sample size (up to 10%), overfitting is almost negligible, attaining ~82% accuracy, precision and recall.

Also note that while accuracy can be calculated as nº of correctly predicted samples/nº of samples, precision, recall and F1 are computed as the weighted average for each class.


Ockham's razor suggests that TF-IDF Vectorizer + Naive Bayes already offers a good performance and should be used given (my) current computational resources. Such a simple model, however, is not expected to generalize as well as a pre-trained llm to new vocabulary or categories. Note that the former one ignores word order, meaning and context.


Storage of models is another element to consider. At least, using my approach, I have decided to serialize the tf-idf + Naive Bayes in a pickle format. This grows massively with the corpus. The BERT model can be more easily stored.


The unit tests contained in the test folder provide 95% coverage of the code, ensuring robustness.

## Run the app locally

Clone the repository and create an image out of the Dockerfile:

```bash
docker build -t amazon_text_classifier .
```

The program receives two parammeters passed as flags:

* command = "train" or "api" if the user wishes to train a new model or perform inverence via the API endpoint
* model = "tfidf" or "llm" to use TfidfVectorizer + Naive-Bayes or a Large Language Model (prajjwal1/bert-tiny)


To setup and run a container we may run:

```bash
docker run -e COMMAND=api -e MODEL=tfidf -it -p 4001:4001 amazon_text_classifier
```

Additionally, we may pass specific configuration of tranining/inference should we wish to. This is done via a config.yaml (there is already one in the image that acts a default file). For example:

```bash
docker run -e COMMAND=api -e MODEL=tfidf -it -p 4001:4001 -v /path/to/local/config.yaml:/app/scripts/config.yaml amazon_text_classifier
```


## Further comments

- What if we were interested in predicting all categories?

This would be a multi-label classification problem. Now, each entity may be assigned two or more labels (categories) and the field category would be the target variable. We can use MultiLabelBinarizer as follows in this example:



```python
from sklearn.preprocessing import MultiLabelBinarizer

# List of categories of three producti
categories = [
    ["Automotive", "Replacement Parts", "Shocks, Struts & Suspension"],
    ["Electronics", "Audio", "Headphones"],
    ["Automotive", "Replacement Parts"],
]

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(categories)
print(y)

```

```bash
[[1 1 1 0 0 0]
 [0 0 0 1 1 1]
 [1 1 0 0 0 0]]
```

A common approach to handle this is OneVsRestClassifier from sklearn. This method fits a classifier for each class (1 if the entity belongs to the class, and 0 otherwise)

In the case of large language models, althoguh they may receive a tensor as the one printed above, we must modify the loss functions. Instead of using cross entropy, where classes are assumed to be **mutually exclusive**, binary cross entropy with logits calculates the loss value for each class independently.


- How would you deploy this API on the cloud?

Steps are given for AWS, albeit they are similar for other cloud providers. Some provider may offer interesing ways of managing containers.

First, we would need to create an image (docker build as explained above), say amazon_image_classifier.
Amazon has got a repository for containers, called AWS Elastic Container Registry (ECR). To authenticate in ECR,


```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
```
We need to tag and push the Docker image, indicating out aws accound id and the region:


```bash
aws ecr create-repository --repository-name amazon_image_classifier
docker tag my-api-image:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/amazon_image_classifier:latest
docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/amazon_image_classifier:latest

```

Then, we need to set up an instance. Either pick individual EC2 instances, should we want to manage virtual machines ourselves, or the ECS approach on EC2 instances. The latter provides an automatic and scalable management of containers. Furthermore, we may consinder ECS Fargate - a serverless computing enginer - if variable API traffic is expected.

Finally, we would have to create a task in the ECS console, add the container, the port and set up a service to run these tasks (for example 1, for testing). We might want to expose our API to the Internet.


- If this model was deployed to categorize products without any supervision which metrics would you check to detect data drifting? When would you need to retrain?

Assuming a non-supervised model, we may have a look at certain elements to spot data drift.
First, note that we are working with textual data. Features distribution can be analysed by means of word embeddings (Word2Vec, BERT, etc.). As we map text into its n-dimensional vector representation, numerical assessments such us cosine similarity, Euclidean distance, Kolmogorov-Smirnov tests can come in handy. If new vocabulary and terms appear, their embeddings will change. Latent Dirichlet Allocation (LDA) can be also used to detect changes in topic modelling

In a more simple approach, regarding models that rely on a corpus (tf-idf, bag of words, etc.), one might detect data drift as new words arise or the frequency they appear in the text with changes. A Chi-Square test can be used here to compare different groups.

Also, variations in the clustering yielded by the model are a sign of data drift. We can compare the populations of two inferences to see if the distribution of predicted categories has been altered. For this, may take the population stability index (PSI) for each class as a metric.

Essentially, the decision to retrain is completely dependent upon the use case. Ideally, the data scientist and the AI engineer should establish a system of alerts - triggered by thresholds - whenever one or some of the aforementioned drift metrics appears to be dangerously high.

