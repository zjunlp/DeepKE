## Task Object

The task objective is to extract specified types of entities and relationships from a given text based on user-provided instructions, for the purpose of constructing a knowledge graph.

Here is an example of a **Knowledge Graph Construction Task**. The user provides a piece of text, referred to as the input, and an instruction that includes the desired types of entities or relationships to be extracted. The system's task is to output all the relationship triples contained in the input and return them in the format specified in the instruction (in this case, in the format of (head entity, relation, tail entity)).

```
instruction="You are an expert specifically trained in extracting relation triples. Given the candidate relation list: ['achievement', 'alternative name', 'area', 'creation time', 'creator', 'event', 'height', 'length', 'located in', 'named after', 'width'], please extract the possible head and tail entities from the input below based on the relation list, and provide the corresponding relation triple. If a certain relation does not exist, output NAN. Please answer in the format of (Subject,Relation,Object)\n."
input="Wewak Airport, also known as Boram Airport, is an airport located in Wewak, Papua New Guinea. IATA code: WWK; ICAO code: AYWK."
output="(Wewak Airport,located in,Wewak)\n(Wewak,located in,Papua New Guinea)\n(Wewak Airport,alternative name,Boram Airport)\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN", "cate": "Building"
```



### 2.1Information Extraction Template
The `template` is used to construct an `instruction` for the input of model, consisting of three parts:
1. **Task Description**: Clearly define the model's function and the task it needs to complete, such as entity recognition, relation extraction, event extraction, etc.
2. **Candidate Label List {s_schema} (optional)**: Define the categories of labels that the model needs to extract, such as entity types, relation types, event types, etc.
3. **Structured Output Format {s_format}**: Specify how the model should present the structured information it extracts.


Template **with specified list of candidate labels**:
```
Named Entity Recognition(NER): You are an expert specialized in entity extraction. With the candidate entity types list: {s_schema}, please extract possible entities from the input below, outputting NAN if a certain entity does not exist. Respond in the format {s_format}.

Relation Extraction(RE): You are an expert in extracting relation triples. With the candidate relation list: {s_schema}, please extract the possible head entities and tail entities from the input below and provide the corresponding relation triples. If a relation does not exist, output NAN. Please answer in the {s_format} format.

Event Extraction(EE): You are a specialist in event extraction. Given the candidate event dictionary: {s_schema}, please extract any possible events from the input below. If an event does not exist, output NAN. Please answer in the format of {s_format}.

Event Type Extraction(EET): As an event analysis specialist, you need to review the input and determine possible events based on the event type directory: {s_schema}. All answers should be based on the {s_format} format. If the event type does not match, please mark with NAN.

Event Argument Extraction(EEA): You are an expert in event argument extraction. Given the event dictionary: {s_schema1}, and the event type and trigger words: {s_schema2}, please extract possible arguments from the following input. If an event argument does not exist, output NAN. Please respond in the {s_format} format.
```


<details>
    <summary><b>Template without specifying a list of candidate labels</b></summary>


  ```
  Named Entity Recognition(NER): Analyze the text content and extract the clear entities. Present your findings in the {s_format} format, skipping any ambiguous or uncertain parts.

  Relation Extraction(RE): Please extract all the relation triples from the text and present the results in the format of {s_format}. Ignore those entities that do not conform to the standard relation template.

  Event Extraction(EE): Please analyze the following text, extract all identifiable events, and present them in the specified format {s_format}. If certain information does not constitute an event, simply skip it.

  Event Type Extraction(EET): Examine the following text content and extract any events you deem significant. Provide your findings in the {s_format} format.

  Event Argument Extraction(EEA): Please extract possible arguments based on the event type and trigger word {s_schema2} from the input below. Answer in the format of {s_format}.
  ```
</details>


<details>
    <summary><b>Candidate Labels {s_schema}</b></summary>


    ```
    NER(Ontonotes): ["date", "organization", "person", "geographical social political", "national religious political", "facility", "cardinal", "location", "work of art", ...]
    RE(NYT): ["ethnicity", "place lived", "geographic distribution", "company industry", "country of administrative divisions", "administrative division of country", ...]
    EE(ACE2005): {"declare bankruptcy": ["organization"], "transfer ownership": ["artifact", "place", "seller", "buyer", "beneficiary"], "marry": ["person", "place"], ...}
    EET(GENIA): ["cell type", "cell line", "protein", "RNA", "DNA"]
    EEA(ACE2005): {"declare bankruptcy": ["organization"], "transfer ownership": ["artifact", "place", "seller", "buyer", "beneficiary"], "marry": ["person", "place"], ...}
    ```
</details>

Here [schema](./kg2instruction/convert/utils.py) provides 12 **text topics** and common relationship types under the topic.

<details>
    <summary><b>Structural Output Format {s_format}</b></summary>


    ```
    Named Entity Recognition(NER): (Entity,Entity Type)

    Relation Extraction(RE): (Subject,Relation,Object)

    Event Extraction(EE): (Event Trigger,Event Type,Argument1#Argument Role1;Argument2#Argument Role2)

    Event Type Extraction(EET): (Event Trigger,Event Type)

    Event Argument Extraction(EEA): (Event Trigger,Event Type,Argument1#Argument Role1;Argument2#Argument Role2)
    ```

</details>


For a more comprehensive understanding of the templates, please refer to the files [ner_converter.py](./kg2instruction/convert/converter/ner_converter.py)、[re_converter.py](./kg2instruction/convert/converter/re_converter.py)、[ee_converter.py](./kg2instruction/convert/converter/ee_converter.py)、[eet_converter.py](./kg2instruction/convert/converter/eet_converter.py)、[eea_converter.py](./kg2instruction/convert/converter/eea_converter.py) and [configs](./configs).




### Data Preprocessing

**Training Data Transformation**

Before inputting data into the model, it needs to be formatted to include `instruction` and `input` fields. To assist with this, we offer a script [kg2instruction/convert.py](./kg2instruction/convert.py), which can batch convert data into a format directly usable by the model.

> Before using the [kg2instruction/convert.py](./kg2instruction/convert.py) script, please ensure you have referred to the [data](./data) directory. Please consult `sample.json` to understand the format of the data before conversion, `schema.json` illustrates the organization of the schema, and `processed.json` describes the format of the data after conversion.


```bash              
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \      # Specifies the language for the conversion script and template, options are ['zh', 'en']
  --task NER \         # Specifies the task type: one of ['RE', 'NER', 'EE', 'EET', 'EEA']
  --sample -1 \        # If -1, randomly samples one of 20 instruction and 4 output formats; if a specific number, uses the corresponding instruction format, range is -1<=sample<20
  --neg_ratio 1 \      # Set the negative sampling ratio for all samples; 1 indicates negative sampling for all samples.
  --neg_schema 1 \     # Set the negative sampling ratio from the schema; 1 indicates embedding the entire schema into the command.
  --random_sort        # Determines whether to randomly sort the list of schemas in the instruction
```

**Negative Sampling**: Assuming dataset A contains labels [a, b, c, d, e, f], for a given sample s, it might involve only labels a and b. Our objective is to randomly introduce some relationships from the candidate relationship list that were originally unrelated to s, such as c and d. However, it's worth noting that in the output, the labels for c and d either won't be included, or they will be output as `NAN`.

`schema_path` is used to specify a schema file (in JSON format) containing three lines of JSON strings. Each line is organized in a fixed format and provides information for named entity recognition (NER) tasks. Taking NER tasks as an example, the meaning of each line is explained as follows:

```
["BookTitle", "Address", "Movie", ...]  # List of entity types
[]  # Empty list
{}  # Empty dictionary
```


<details>
  <summary><b>More</b></summary>



```
For Relation Extraction (RE) tasks:
[]                                                 # Empty list
["Founder", "Number", "RegisteredCapital", ...]    # List of relation types
{}                                                 # Empty dictionary


For Event Extraction (EE) tasks:
["Social Interaction-Thanks", "Organizational Action-OpeningCeremony", "Competition Action-Withdrawal", ...]        # List of event types
["DismissingParty", "TerminatingParty", "Reporter", "ArrestedPerson"]       # List of argument roles
{"OrganizationalRelation-Layoff": ["LayoffParty", "NumberLaidOff", "Time"], "LegalAction-Sue": ["Plaintiff", "Defendant", "Time"], ...}         # Dictionary of event types


For Event Type Extraction(EET) tasks:
["Social Interaction-Thanks", "Organizational Action-OpeningCeremony", "Competition Action-Withdrawal", ...]         # List of event types
[]                               # Empty list
{}                               # Empty dictionary


For Event Argument Extraction(EEA) tasks:
["Social Interaction-Thanks", "Organizational Action-OpeningCeremony", "Competition Action-Withdrawal", ...]                  # List of event types
["DismissingParty", "TerminatingParty", "Reporter", "ArrestedPerson"]           # List of argument roles
{"OrganizationalRelation-Layoff": ["LayoffParty", "NumberLaidOff", "Time"], "LegalAction-Sue": ["Plaintiff", "Defendant", "Time"], ...}             # Dictionary of event types
```

</details>

For more detailed information on the schema file, you can refer to the `schema.json` file in the respective task directories under the [data](./data) directory.


**Testing Data Transformation**

For test data, you can use the [kg2instruction/convert_test.py](./kg2instruction/convert_test.py) script, which does not require the data to contain label fields (`entity`, `relation`, `event`), just the input field and the corresponding schema_path.

```bash
python kg2instruction/convert_test.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \
  --task NER \
  --sample 0 \
  --schema_num 4     # For whether to segment a single data into a schema, if there are 16 schema labels, each data after segmentation corresponds to 4 test data, distinguished by a 'split' field
```


**Data Transformation Examples**

Here is an example of data conversion for Named Entity Recognition (NER) task:

```
Before Transformation:
{
    "input": "In contrast, the rain-soaked battle between Qingdao Sea Bulls and Guangzhou Songri Team, although also ended in a 0:0 draw, was uneventful.",
    "entity": [{"entity": "Guangzhou Songri Team", "entity_type": "Organizational Structure"}, {"entity": "Qingdao Sea Bulls", "entity_type": "Organizational Structure"}]
}

After Transformation:
{
    "id": "e88d2b42f8ca14af1b77474fcb18671ed3cacc0c75cf91f63375e966574bd187",
    "instruction": "Please identify and list the entity types mentioned in the given text ['Organizational Structure', 'Person', 'Geographical Location']. If a type doesn't exist, please indicate it as NAN. Provide your answer in the format (entity, entity type).",
    "input": "In contrast, the rain-soaked battle between Qingdao Sea Bulls and Guangzhou Songri Team, although also ended in a 0:0 draw, was uneventful.",
    "output": "(Qingdao Sea Bulls,Organizational Structure)\n(Guangzhou Songri Team,Organizational Structure)\nNAN\nNAN"
}
```

Before conversion, the data format needs to adhere to the structure specified in the `DeepKE/example/llm/InstructKGC/data` directory for each task (such as NER, RE, EE). Taking NER task as an example, the input text should be labeled as the `input` field, and the annotated data should be labeled as the `entity` field, which is a list of dictionaries containing multiple key-value pairs for `entity` and `entity_type`.

After data conversion, you will obtain structured data containing the `input` text, `instruction` (providing detailed instructions about candidate entity types ['Organization', 'Person', 'Location'] and the expected output format), and `output` (listing all entity information recognized in the `input` in the form of (entity, entity type)).



## Model Output Conversion & F1 Calculation
We provide a script, [evaluate.py](./kg2instruction/evaluate.py), to convert the model's string outputs into lists and calculate the F1 score.

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task NER \
  --language zh
```