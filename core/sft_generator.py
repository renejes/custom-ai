"""
SFT (Supervised Fine-Tuning) data generation module.
Supports OpenRouter (cloud) and Ollama (local) for generating training data.
"""

import aiohttp
import asyncio
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path


class SFTGenerator:
    """Generates SFT training data using OpenRouter or Ollama."""

    def __init__(self, provider: str, api_key: str = "", base_url: str = "http://localhost:11434"):
        """
        Initialize SFT Generator.

        Args:
            provider: "openrouter" or "ollama"
            api_key: API key (required for OpenRouter)
            base_url: Base URL for Ollama
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

        # Default prompts
        self.default_system_prompt = """You are an expert educator creating training data for an AI assistant.
Generate realistic and educational question-answer pairs.
Always respond with valid JSON in this exact format:
{"instruction": "the question", "output": "the detailed answer"}"""

        self.default_user_prompt = """Generate 1 educational question-answer pair about {topic}.
Respond ONLY with JSON in this format:
{{"instruction": "question here", "output": "detailed answer here"}}"""

    async def test_connection(self, model: str) -> Tuple[bool, str]:
        """
        Test API connection.

        Args:
            model: Model name to test

        Returns:
            Tuple of (success, message)
        """
        try:
            if self.provider == "openrouter":
                return await self._test_openrouter(model)
            elif self.provider == "ollama":
                return await self._test_ollama(model)
            else:
                return False, f"Unknown provider: {self.provider}"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    async def _test_openrouter(self, model: str) -> Tuple[bool, str]:
        """Test OpenRouter connection."""
        if not self.api_key:
            return False, "API key is required for OpenRouter"

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        return True, " OpenRouter connection successful"
                    elif resp.status == 401:
                        return False, "L Invalid API key"
                    elif resp.status == 402:
                        return False, "L Insufficient credits"
                    else:
                        text = await resp.text()
                        return False, f"L Error {resp.status}: {text[:100]}"
        except asyncio.TimeoutError:
            return False, "L Connection timeout"
        except Exception as e:
            return False, f"L Error: {str(e)}"

    async def _test_ollama(self, model: str) -> Tuple[bool, str]:
        """Test Ollama connection."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": "Hi",
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        return True, " Ollama connection successful"
                    elif resp.status == 404:
                        return False, f"L Model '{model}' not found. Run: ollama pull {model}"
                    else:
                        text = await resp.text()
                        return False, f"L Error {resp.status}: {text[:100]}"
        except aiohttp.ClientConnectorError:
            return False, "L Cannot connect to Ollama. Is it running?"
        except asyncio.TimeoutError:
            return False, "L Connection timeout"
        except Exception as e:
            return False, f"L Error: {str(e)}"

    async def generate_single_sample(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7
    ) -> Tuple[bool, str, str]:
        """
        Generate a single sample (for custom Q&A generation).

        Args:
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt (already formatted with chunk content)
            temperature: Sampling temperature

        Returns:
            Tuple of (success, response_text, error_message)
        """
        try:
            if self.provider == "openrouter":
                result = await self._generate_openrouter_raw(
                    model, system_prompt, user_prompt, temperature
                )
            elif self.provider == "ollama":
                result = await self._generate_ollama_raw(
                    model, system_prompt, user_prompt, temperature
                )
            else:
                return False, "", f"Unknown provider: {self.provider}"

            if result:
                return True, result, ""
            else:
                return False, "", "No response generated"

        except Exception as e:
            return False, "", str(e)

    async def generate_samples(
        self,
        model: str,
        num_samples: int,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float = 0.7,
        topic: str = "general education",
        progress_callback=None
    ) -> Tuple[bool, List[Dict], str]:
        """
        Generate multiple SFT samples.

        Args:
            model: Model name
            num_samples: Number of samples to generate
            system_prompt: System prompt
            user_prompt_template: User prompt template (can use {topic})
            temperature: Sampling temperature
            topic: Topic for generation
            progress_callback: Optional callback function(current, total, sample)

        Returns:
            Tuple of (success, samples_list, error_message)
        """
        samples = []
        errors = []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)

        async def generate_one(index: int):
            """Generate one sample."""
            async with semaphore:
                try:
                    # Format user prompt
                    user_prompt = user_prompt_template.replace("{topic}", topic)

                    # Generate
                    if self.provider == "openrouter":
                        sample = await self._generate_openrouter(
                            model, system_prompt, user_prompt, temperature
                        )
                    elif self.provider == "ollama":
                        sample = await self._generate_ollama(
                            model, system_prompt, user_prompt, temperature
                        )
                    else:
                        return None

                    if sample:
                        samples.append(sample)
                        if progress_callback:
                            await progress_callback(len(samples), num_samples, sample)

                    return sample

                except Exception as e:
                    errors.append(f"Sample {index + 1}: {str(e)}")
                    return None

        # Generate all samples
        tasks = [generate_one(i) for i in range(num_samples)]
        await asyncio.gather(*tasks)

        if not samples:
            error_msg = "\n".join(errors) if errors else "No samples generated"
            return False, [], error_msg

        return True, samples, ""

    async def _generate_openrouter_raw(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> Optional[str]:
        """Generate raw text via OpenRouter."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 500
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
                    if resp.status != 200:
                        return None

                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content

        except Exception as e:
            print(f"OpenRouter error: {e}")
            return None

    async def _generate_openrouter(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> Optional[Dict]:
        """Generate one sample via OpenRouter."""
        content = await self._generate_openrouter_raw(model, system_prompt, user_prompt, temperature)
        if content:
            return self._parse_response(content)
        return None

    async def _generate_ollama_raw(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> Optional[str]:
        """Generate raw text via Ollama."""
        url = f"{self.base_url}/api/generate"

        # Combine prompts for Ollama
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        return None

                    data = await resp.json()
                    content = data.get("response", "")
                    return content

        except Exception as e:
            print(f"Ollama error: {e}")
            return None

    async def _generate_ollama(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> Optional[Dict]:
        """Generate one sample via Ollama."""
        content = await self._generate_ollama_raw(model, system_prompt, user_prompt, temperature)
        if content:
            return self._parse_response(content)
        return None

    @staticmethod
    def _parse_response(content: str) -> Optional[Dict]:
        """
        Parse LLM response to extract JSON.

        Args:
            content: Raw response content

        Returns:
            Parsed sample dict or None
        """
        try:
            # Try direct JSON parse
            sample = json.loads(content)

            # Validate structure
            if "instruction" in sample and "output" in sample:
                return {
                    "instruction": sample["instruction"],
                    "output": sample["output"]
                }

        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
                try:
                    sample = json.loads(json_str)
                    if "instruction" in sample and "output" in sample:
                        return {
                            "instruction": sample["instruction"],
                            "output": sample["output"]
                        }
                except:
                    pass

            # Try to extract JSON without markdown
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                try:
                    sample = json.loads(json_str)
                    if "instruction" in sample and "output" in sample:
                        return {
                            "instruction": sample["instruction"],
                            "output": sample["output"]
                        }
                except:
                    pass

        return None

    def save_samples(self, samples: List[Dict], output_path: str) -> Tuple[bool, str]:
        """
        Save samples to JSONL file.

        Args:
            samples: List of sample dicts
            output_path: Path to output file

        Returns:
            Tuple of (success, message)
        """
        try:
            # Create directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Write JSONL
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            return True, f" Saved {len(samples)} samples to {output_path}"

        except Exception as e:
            return False, f"L Error saving samples: {str(e)}"

    def load_samples(self, input_path: str) -> Tuple[bool, List[Dict], str]:
        """
        Load samples from JSONL file.

        Args:
            input_path: Path to input file

        Returns:
            Tuple of (success, samples, error_message)
        """
        try:
            samples = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        if "instruction" in sample and "output" in sample:
                            samples.append(sample)

            return True, samples, ""

        except Exception as e:
            return False, [], f"Error loading samples: {str(e)}"


if __name__ == "__main__":
    # Test SFT generator
    async def test():
        # Test Ollama
        generator = SFTGenerator(provider="ollama")

        print("Testing Ollama connection...")
        success, msg = await generator.test_connection("llama3:8b")
        print(msg)

        if success:
            print("\nGenerating sample...")
            success, samples, error = await generator.generate_samples(
                model="llama3:8b",
                num_samples=1,
                system_prompt=generator.default_system_prompt,
                user_prompt_template=generator.default_user_prompt,
                topic="mathematics"
            )

            if success and samples:
                print(f"\nGenerated {len(samples)} sample(s):")
                print(json.dumps(samples[0], indent=2))

    asyncio.run(test())
