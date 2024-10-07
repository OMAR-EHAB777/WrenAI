
# Pipelines Documentation
![PipeLines Layer Architecture](/docs/Pipelines_Layer_Architechture.drawio.png)
This folder contains various pipelines used in the system, which are responsible for handling SQL explanation, generation, and retrieval tasks. Each pipeline is designed to interact with the underlying database schema and provide necessary transformations, validations, and SQL executions. The major pipelines are described below:

## 1. SQL Explanation Pipeline
- **Components**: `SQLAnalysisPreprocessor`, `SQLExplanationGenerationPostProcessor`
- **Description**: This pipeline explains SQL queries by analyzing their structure and returning easy-to-understand explanations of the SQL logic, leveraging structured results for complex SQL queries.
- **Key Features**:
  - Breakdown of SQL queries into simpler explanations.
  - Handles SQL constructs such as `SELECT`, `GROUP BY`, `JOIN`, etc.

## 2. SQL Generation Pipeline
- **Components**: `PromptBuilder`, `SQLGenPostProcessor`
- **Description**: This pipeline takes natural language questions and generates SQL queries. It follows strict SQL generation rules to ensure valid SQL is generated for each request.
- **Key Features**:
  - Generates SQL for user input.
  - Applies strict rules for generating correct SQL syntax.
  - Supports calculated fields and metrics in SQL schema.

## 3. SQL Retrieval Pipeline
- **Components**: `ScoreFilter`, `OutputFormatter`, `DocumentStore`, `Embedder`
- **Description**: This pipeline retrieves relevant SQL queries and data from historical questions and views based on embeddings.
- **Key Features**:
  - Retrieves past queries based on semantic similarity.
  - Filters results based on a score threshold.

## 4. Common Utilities
- **Files**: `common.py`
- **Description**: Contains shared utility components such as post-processors (`SQLGenPostProcessor`, `SQLBreakdownGenPostProcessor`) that are used to validate and format SQL responses.
- **Key Features**:
  - Cleans and validates SQL generation responses.
  - Adds error handling and debugging for invalid SQL.

## 5. Retrival Pipelines
- **Files**: `historical_question.py`, `retrival.py`
- **Description**: Manages retrieval of historical queries and constructs retrieval results from the database schema.
- **Key Features**:
  - Embeds user queries and retrieves relevant historical data.
  - Constructs tables and views based on retrieval results.

## How to Use
- Each pipeline class provides a `run` method to execute the entire pipeline and return results.
- To visualize the pipeline steps, use the `visualize` method available in each pipeline class.

