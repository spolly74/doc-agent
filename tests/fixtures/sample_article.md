---
title: "Guardrails and controls overview in Microsoft Foundry"
description: "Learn how guardrails and controls work in Microsoft Foundry to implement safety measures for your models and agents."
ms.topic: conceptual
ms.date: 11/05/2025
ms.author: ssalgado
ms.service: azure-ai-foundry
ms.custom:
  - azure-ai-guardrails
  - build-2025
customer-intent: As a developer, I want to understand how guardrails work in Microsoft Foundry so that I can implement appropriate safety measures for my models and agents.
---

# Guardrails and controls overview in Microsoft Foundry

Guardrails are a key feature of Microsoft Foundry that help you implement safety measures for your AI models and agents. They define what types of content should be detected and how the system should respond when problematic content is identified.

Controls define the specific rules within a guardrail. Each control specifies a risk category, the detection level, and the action to take when that risk is detected.

## Prerequisites

Before you begin, make sure you have:

- An Azure subscription
- Access to Microsoft Foundry
- Basic understanding of AI safety concepts

## Guardrails for agents vs models

The following table shows how guardrails apply differently to agents and models:

| Feature | Agents | Models |
|---------|--------|--------|
| Default guardrails | Yes | Yes |
| Custom guardrails | Yes | Limited |
| Runtime override | Yes | No |

## Configuring guardrails

You can configure guardrails using the SDK or the portal. Here's a basic example:

```python
from foundry import GuardrailConfig

config = GuardrailConfig(
    name="my-guardrail",
    controls=[
        {"risk": "hate_speech", "level": "high", "action": "block"},
        {"risk": "violence", "level": "medium", "action": "warn"},
    ]
)
```

After configuring, you can apply the guardrail to your agent:

```python
agent.apply_guardrail(config)
```

## Default safety policy

All models and agents come with a default safety policy that provides baseline protection. You can customize this policy to fit your specific needs.

> [!NOTE]
> The default policy cannot be completely disabled, only customized.

## Next steps

- [Create custom guardrails](./how-to-create-guardrails.md)
- [Guardrails API reference](./api-reference/guardrails.md)
- [Safety best practices](./safety-best-practices.md)
