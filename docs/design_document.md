# `yaaal` - Yet Another Agentic AI Library

- [`yaaal` - Yet Another Agentic AI Library](#yaaal---yet-another-agentic-ai-library)
  - [1. Project Overview](#1-project-overview)
    - [Primary Purpose](#primary-purpose)
    - [1.1 Target Audience](#11-target-audience)
    - [1.2 Problem Statement](#12-problem-statement)
    - [1.3 Unique Selling Points](#13-unique-selling-points)
  - [2. Goals and Objectives](#2-goals-and-objectives)
    - [2.1 Core Goals](#21-core-goals)
  - [3. Key Features](#3-key-features)
    - [3.1 Message and Conversation Model](#31-message-and-conversation-model)
    - [3.2 Message Templates](#32-message-templates)
    - [3.3 Prompt Creation](#33-prompt-creation)
    - [3.4 Separation of Prompt and LLM API](#34-separation-of-prompt-and-llm-api)
    - [3.4 Tool Use](#34-tool-use)
  - [4. Architecture and Design](#4-architecture-and-design)
  - [4.1 Overview of Architecture](#41-overview-of-architecture)
    - [4.2 Design Patterns and Principles](#42-design-patterns-and-principles)
    - [4.4 Dependencies and Compatibility](#44-dependencies-and-compatibility)
  - [5. Licensing and Contribution](#5-licensing-and-contribution)
    - [5.1 License](#51-license)
    - [5.2 Contributing](#52-contributing)
  - [6. Open Questions and Future Work](#6-open-questions-and-future-work)
    - [State and Context Management](#state-and-context-management)
    - [Call Graphs](#call-graphs)
    - [Default Tools, Agents](#default-tools-agents)
    - [OpenTelemetry Tracing](#opentelemetry-tracing)
    - [Middleware Responsibility](#middleware-responsibility)

## 1. Project Overview

### Primary Purpose

`yaaal` is designed to be a highly composable and lightweight framework for building AI agents with minimal dependencies.
It aims to offer developers a simple yet flexible toolkit for creating autonomous systems that can interact with various environments, make decisions, and perform tasks in a modular manner.

### 1.1 Target Audience

The primary audience for `yaaal` includes developers, AI researchers, and hobbyists who are looking for a simple, modular framework for building intelligent agents without dealing with the overhead of heavy AI libraries or complicated dependencies.

### 1.2 Problem Statement

Creating AI agents often involves tying together a series of tools and components—such as decision-making logic, environment interaction, state management, and more—into a cohesive system.
Many existing frameworks are either too heavy or too rigid, making them difficult to customize or extend for specific use cases.
`yaaal` aims to fill this gap by providing a minimalistic and composable framework where users can easily mix and match different components and build agents with ease.

### 1.3 Unique Selling Points

- Composability: yaaal's design emphasizes flexibility, allowing users to build agents by composing small, reusable modules.
- Minimal Dependencies: Unlike many AI frameworks that require heavy dependencies (like TensorFlow or PyTorch), `yaaal` strives to be lightweight and easy to integrate into any project.
- Simplicity and Extensibility: The library is designed to be easy to use for beginners while also being powerful enough for advanced users who need customizability.

## 2. Goals and Objectives

### 2.1 Core Goals

- Provide a lightweight, composable framework for building AI agents with minimal dependencies.
- Enable seamless interaction with LLM-based APIs and other external tools using a consistent interface.
- Support flexible state management and context passing for agents, going beyond simple message histories.
- Allow easy configuration and management of prompts, messages, and tools in agentic workflows.
- Enable introspection into the agent's execution flow with OpenTelemetry tracing for better visibility into how agents operate.

## 3. Key Features

### 3.1 Message and Conversation Model

**Message**: This class defines a structured message that includes a role (such as system, user, or assistant) and context (the content of the message). It follows common LLM chat API conventions for compatibility.
**Conversation**: A container for multiple Message objects, representing a full interaction history. It includes utility methods for converting the message objects into API-compatible formats.

### 3.2 Message Templates

**MessageTemplate**: Allows users to define standard message formats with placeholders that are filled dynamically. Each template includes the message role and a template string.
Validation: Before rendering a template into a message, optional validation checks ensure that the provided variables meet predefined criteria.

### 3.3 Prompt Creation

**Prompt**: A composite object that uses multiple MessageTemplate objects to create a complete conversation flow. It manages the logic for generating and structuring messages and enables greater flexibility in designing agent interactions.

### 3.4 Separation of Prompt and LLM API

**Callable Interface**: The Caller object makes the Prompt callable by implementing Python's `__call__()` method. It abstracts the execution of prompts, handling the communication with external APIs or services.

**Response Validation**: The Caller can include optional validation logic to ensure the API response is correct before passing it to the agent.

### 3.4 Tool Use

**Tool Abstraction**: Tools are functions that provide specialized functionality, such as making an API call or performing a calculation. `yaaal` makes provisions for use of native python functions as well as Callers as tools.

**Tool Signature**: `yaaal` support using native python functions and Callers as tools.
The `@tool` decorator allows the developer to denote Python functions as callable tools that agents can invoke during their workflows.
Both `@tool` decorated functions and Caller objects have a `signature()` method that describes the tool inputs as a Pydantic BaseModel
for easy conversion to json schema, allowing easy integration into LLM tool-calling requests.

## 4. Architecture and Design

## 4.1 Overview of Architecture

The architecture of `yaaal` is modular and designed for flexibility. The library enables users to create AI agents by combining simple building blocks into a comprehensive agent framework.
Each component--Message, Conversation, MessageTemplate, Prompt, Caller, Agent --is decoupled to allow for easy composition and modification.

### 4.2 Design Patterns and Principles

**Modular and Composable**: `yaaal` is built on the principle of composability. Each component can be independently customized and reused in different agentic workflows.

**Separation of Concerns**: Each component in `yaaal` has a clear, single responsibility (e.g., message generation, prompt handling, API interaction).

**Minimal Dependencies**: The core functionality is lightweight, with optional dependencies for advanced features like OpenTelemetry tracing or response validation.

**Flexibility and Extensibility**: `yaaal` is designed to be highly customizable, with clear interfaces for extending or replacing core components.

### 4.4 Dependencies and Compatibility

**Core Dependencies**: `yaaal` has minimal dependencies but assumes `aisuite` and `pydantic>=2.10` and optionally uses OpenTelemetry for tracing.

**Compatibility**: Works in any standard Python environment (>=3.11) and can be easily integrated with other tools and services. We generally assume we are building for OpenAI API compatibility.

## 5. Licensing and Contribution

### 5.1 License

`yaaal` is  released under the Apache License 2.0 with a Commons Clause Noncommercial Addendum.
This license allows you to freely use, modify, and distribute the software as long as you are not using it to provide a paid service (e.g., in a SaaS or commercial offering) without acquiring a commercial license.
The full terms of the license can be found in the [LICENSE file](../LICENSE) in the repository.

### 5.2 Contributing

Contributions to the `yaaal` project are welcome! You can submit issues, feature requests, and pull requests on the GitHub repository.
By contributing to this project, you agree that your contributions will be licensed under the terms of the Apache License 2.0.
If you have questions about licensing or if you're interested in acquiring a commercial license, please contact the project maintainers through the GitHub repository.

## 6. Open Questions and Future Work

### State and Context Management

**Context**: `yaaal` supports context objects that store additional state, such as user preferences, environmental data, or previous tool outputs, beyond the message history.
This allows agents to maintain continuity across interactions and makes it easier to manage complex, stateful workflows.

### Call Graphs

**Call Chains**: Callers can be chained together to form linear, modular workflows. A Caller can call another Caller, passing the result along in a sequence of operations.
This enables step-by-step execution of tasks, where each Caller handles a distinct part of the process. For example, one Caller could generate text, another could refine it, and a final Caller could validate the output.

**Agent-to-Agent Communication**: `yaaal` allows Agents to nest within each other, enabling an Agent to call another Agent as part of its decision-making process.
This allows for hierarchical agent structures, where one agent's output can be used as input for another agent.
For example, an "AssistantAgent" could call a "SummarizerAgent" to summarize a document before making further decisions based on the summary.

**Modular Workflow Composition**: The ability to nest Callers and Agents promotes a high degree of modularity.
Each agent or caller can focus on a specific task or subtask, and users can build sophisticated workflows by composing agents and callers in chains or hierarchies. This modular approach helps with reusability, testing, and scalability.

### Default Tools, Agents

**Default Tools**: Should `yaaal` provide default tools such as web search, URL extraction (perhaps via Jina Reader?), etc?

**Default Agents**: Should `yaaal` provide default Agents, such as ReAct, or a mechanism to automatically select a subset of appropriate tools?

### OpenTelemetry Tracing

**Tracing Integration**: OpenTelemetry tracing is integrated into the framework to track the agent's actions, API calls, and internal processes.

**Exporting Traces**: Traces can be exported to OpenTelemetry-compatible systems (e.g., Jaeger, Zipkin), enabling detailed introspection and monitoring.

**Trace Contexts**: Trace context is passed along with messages, API calls, and tool executions to ensure a complete view of the agent's operations.

### Middleware Responsibility

Whose responsibility is it to implement a middleware layer? If a dev is creating a chat application with something like Gradio,
do they need to implement the middleware to facilitate the connection to `yaaal` Agents or Callers, or should `yaaal` define its interface middleware?
