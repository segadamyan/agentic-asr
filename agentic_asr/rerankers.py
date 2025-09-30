"""Reranker implementations for improving vector search results."""

import re
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from collections import Counter
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    def __init__(self, name: str = "base"):
        self.name = name
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank search results based on query relevance.
        
        Args:
            query: The search query
            results: List of search results with 'content' and 'similarity_score' keys
            top_k: Maximum number of results to return
            
        Returns:
            Reranked list of results
        """
        pass


class NoReranker(BaseReranker):
    """Identity reranker that returns results unchanged."""
    
    def __init__(self):
        super().__init__("no_reranker")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return results unchanged."""
        if top_k is not None:
            return results[:top_k]
        return results


class BM25Reranker(BaseReranker):
    """BM25-based reranker for improving search results."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        super().__init__("bm25")
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _compute_bm25_score(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
        avg_doc_length: float,
        corpus_size: int,
        term_doc_freq: Dict[str, int]
    ) -> float:
        """Compute BM25 score for a document."""
        doc_length = len(doc_tokens)
        doc_token_counts = Counter(doc_tokens)
        score = 0.0
        
        for term in query_tokens:
            if term in doc_token_counts:
                tf = doc_token_counts[term]  # Term frequency in document
                df = term_doc_freq.get(term, 0)  # Document frequency in corpus
                
                if df > 0:
                    # IDF component
                    idf = math.log((corpus_size - df + 0.5) / (df + 0.5))
                    
                    # TF component with saturation and length normalization
                    tf_component = (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                    )
                    
                    score += idf * tf_component
        
        return score
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank results using BM25 scoring."""
        if not results:
            return results
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return results[:top_k] if top_k else results
        
        # Tokenize all documents and build term statistics
        doc_tokens_list = []
        term_doc_freq = Counter()
        total_length = 0
        
        for result in results:
            doc_tokens = self._tokenize(result['content'])
            doc_tokens_list.append(doc_tokens)
            total_length += len(doc_tokens)
            
            # Count documents containing each term
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                term_doc_freq[term] += 1
        
        avg_doc_length = total_length / len(results) if results else 0
        corpus_size = len(results)
        
        # Compute BM25 scores
        bm25_scores = []
        for i, result in enumerate(results):
            bm25_score = self._compute_bm25_score(
                query_tokens,
                doc_tokens_list[i],
                avg_doc_length,
                corpus_size,
                term_doc_freq
            )
            bm25_scores.append(bm25_score)
        
        # Combine with original similarity scores
        reranked_results = []
        for i, result in enumerate(results):
            new_result = result.copy()
            new_result['bm25_score'] = bm25_scores[i]
            # Combine scores (you can adjust the weighting)
            new_result['combined_score'] = (
                0.7 * result.get('similarity_score', 0) + 
                0.3 * bm25_scores[i]
            )
            reranked_results.append(new_result)
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.debug(f"BM25 reranker processed {len(results)} results")
        
        if top_k is not None:
            return reranked_results[:top_k]
        return reranked_results


class KeywordBoostReranker(BaseReranker):
    """Reranker that boosts results containing exact keyword matches."""
    
    def __init__(self, boost_factor: float = 1.5, case_sensitive: bool = False):
        super().__init__("keyword_boost")
        self.boost_factor = boost_factor
        self.case_sensitive = case_sensitive
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction - split on whitespace and remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        if not self.case_sensitive:
            query = query.lower()
        
        keywords = []
        for word in re.findall(r'\b\w+\b', query):
            if word.lower() not in stop_words and len(word) > 2:
                keywords.append(word)
        
        return keywords
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank results by boosting exact keyword matches."""
        if not results:
            return results
        
        keywords = self._extract_keywords(query)
        if not keywords:
            return results[:top_k] if top_k else results
        
        reranked_results = []
        for result in results:
            new_result = result.copy()
            content = result['content']
            
            if not self.case_sensitive:
                content = content.lower()
            
            # Count keyword matches
            keyword_matches = 0
            for keyword in keywords:
                keyword_matches += content.count(keyword)
            
            # Apply boost based on keyword matches
            boost = 1.0 + (keyword_matches * (self.boost_factor - 1.0))
            
            new_result['keyword_matches'] = keyword_matches
            new_result['keyword_boost'] = boost
            new_result['boosted_score'] = result.get('similarity_score', 0) * boost
            
            reranked_results.append(new_result)
        
        # Sort by boosted score
        reranked_results.sort(key=lambda x: x['boosted_score'], reverse=True)
        
        logger.debug(f"Keyword boost reranker processed {len(results)} results")
        
        if top_k is not None:
            return reranked_results[:top_k]
        return reranked_results


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker for more accurate semantic scoring."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32
    ):
        super().__init__("cross_encoder")
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            # Fall back to a smaller model that might work better with Armenian
            fallback_model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
            try:
                self.model = CrossEncoder(fallback_model, max_length=self.max_length)
                self.model_name = fallback_model
                logger.info(f"Loaded fallback cross-encoder model: {fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise e2
    
    def _prepare_pairs(self, query: str, results: List[Dict[str, Any]]) -> List[List[str]]:
        """Prepare query-document pairs for cross-encoder."""
        pairs = []
        for result in results:
            # Truncate content to fit within model's max length
            content = result['content']
            if len(content) > self.max_length - len(query) - 10:  # Leave room for special tokens
                content = content[:self.max_length - len(query) - 10]
            
            pairs.append([query, content])
        
        return pairs
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder model."""
        if not results or self.model is None:
            return results[:top_k] if top_k else results
        
        try:
            # Prepare query-document pairs
            pairs = self._prepare_pairs(query, results)
            
            # Get cross-encoder scores in batches
            all_scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores.tolist())
            
            # Create reranked results
            reranked_results = []
            for i, result in enumerate(results):
                new_result = result.copy()
                new_result['cross_encoder_score'] = float(all_scores[i])
                
                # Combine with original similarity score (adjustable weighting)
                original_score = result.get('similarity_score', 0)
                combined_score = 0.4 * original_score + 0.6 * all_scores[i]
                new_result['combined_score'] = combined_score
                
                reranked_results.append(new_result)
            
            # Sort by combined score
            reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            logger.debug(f"Cross-encoder reranker processed {len(results)} results")
            
            if top_k is not None:
                return reranked_results[:top_k]
            return reranked_results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}. Returning original results.")
            return results[:top_k] if top_k else results


class MultilingualCrossEncoderReranker(CrossEncoderReranker):
    """Cross-encoder reranker optimized for multilingual content including Armenian."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        max_length: int = 512,
        batch_size: int = 32,
        armenian_boost: float = 1.1
    ):
        """Initialize multilingual cross-encoder reranker.
        
        Args:
            model_name: Multilingual cross-encoder model name
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            armenian_boost: Boost factor for Armenian text detection
        """
        self.armenian_boost = armenian_boost
        super().__init__(model_name, max_length, batch_size)
        self.name = "multilingual_cross_encoder"
    
    def _detect_armenian_text(self, text: str) -> float:
        """Detect Armenian script in text and return ratio."""
        armenian_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                # Armenian Unicode ranges: 0530-058F, FB13-FB17
                if (0x0530 <= ord(char) <= 0x058F) or (0xFB13 <= ord(char) <= 0xFB17):
                    armenian_chars += 1
        
        return armenian_chars / total_chars if total_chars > 0 else 0.0
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank with Armenian language boost."""
        # First apply cross-encoder reranking
        reranked_results = super().rerank(query, results, top_k=None)
        
        # Apply Armenian boost
        query_armenian_ratio = self._detect_armenian_text(query)
        
        for result in reranked_results:
            content_armenian_ratio = self._detect_armenian_text(result['content'])
            
            # Boost if both query and content have Armenian text
            if query_armenian_ratio > 0.1 and content_armenian_ratio > 0.1:
                armenian_boost_factor = 1.0 + (
                    self.armenian_boost - 1.0
                ) * min(query_armenian_ratio, content_armenian_ratio)
                
                result['armenian_boost'] = armenian_boost_factor
                result['combined_score'] = result.get('combined_score', 0) * armenian_boost_factor
            else:
                result['armenian_boost'] = 1.0
        
        # Re-sort after applying Armenian boost
        reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.debug(f"Multilingual cross-encoder reranker processed {len(results)} results with Armenian detection")
        
        if top_k is not None:
            return reranked_results[:top_k]
        return reranked_results


class LLMAgentReranker(BaseReranker):
    """LLM-based reranker that uses an agent to score relevance."""
    
    def __init__(
        self,
        provider_name: str = "anthropic",
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        max_chunks_per_batch: int = 5,
        temperature: float = 0.1
    ):
        super().__init__("llm_agent")
        
        self.provider_name = provider_name
        self.model = model
        self.max_chunks_per_batch = max_chunks_per_batch
        self.temperature = temperature
        
        # Import LLM components
        try:
            from agentic_asr.llm.providers import create_llm_provider
            from agentic_asr.core.models import LLMProviderConfig, GenerationSettings, History, Message, RoleEnum
            self.create_llm_provider = create_llm_provider
            self.LLMProviderConfig = LLMProviderConfig
            self.GenerationSettings = GenerationSettings
            self.History = History
            self.Message = Message
            self.RoleEnum = RoleEnum
        except ImportError as e:
            logger.error(f"Failed to import LLM components: {e}")
            raise ImportError("LLM components required for LLMAgentReranker")
        
        # Get API key from config if not provided
        if not api_key:
            from agentic_asr.config import Config
            if provider_name.lower() == "openai":
                api_key = Config.OPENAI_API_KEY
            elif provider_name.lower() == "anthropic":
                api_key = Config.ANTHROPIC_API_KEY
        
        if not api_key:
            raise ValueError(f"API key required for {provider_name} provider")
        
        # Initialize LLM provider
        self.llm_config = self.LLMProviderConfig(
            provider_name=provider_name,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=1000
        )
        
        self.llm_provider = self.create_llm_provider(self.llm_config)
        self.generation_settings = self.GenerationSettings(
            temperature=temperature,
            max_tokens=500
        )
    
    def _create_scoring_prompt(self, query: str, chunk: str) -> str:
        """Create a prompt for scoring relevance."""
        return f"""You are a relevance scoring agent. Your task is to score how relevant a text chunk is to a given query.

Query: "{query}"

Text Chunk: "{chunk}"

Please analyze the semantic relevance between the query and the text chunk. Consider:
1. Topical similarity and overlap
2. Conceptual relationships
3. Contextual relevance
4. Language compatibility (both Armenian and English are acceptable)

Provide a relevance score from 0.0 to 1.0 where:
- 0.0 = Completely irrelevant
- 0.3 = Slightly related but not very relevant
- 0.5 = Moderately relevant
- 0.7 = Highly relevant
- 1.0 = Extremely relevant and directly answers the query

Respond with only the numerical score (e.g., 0.75). Do not include any explanation."""
    
    async def _score_chunk_async(self, query: str, chunk: str) -> float:
        """Score a single chunk using the LLM."""
        try:
            prompt = self._create_scoring_prompt(query, chunk)
            
            history = self.History(messages=[
                self.Message(
                    role=self.RoleEnum.USER,
                    content=prompt
                )
            ])
            
            response = await self.llm_provider.generate_response(
                history=history,
                settings=self.generation_settings
            )
            
            # Extract numerical score from response
            score_text = response.content.strip()
            
            # Try to extract a number from the response
            import re
            score_match = re.search(r'([0-1]\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            else:
                logger.warning(f"Could not parse score from LLM response: {score_text}")
                return 0.5  # Default score
                
        except Exception as e:
            logger.error(f"Error scoring chunk with LLM: {e}")
            return 0.0
    
    def _score_chunk_sync(self, query: str, chunk: str) -> float:
        """Synchronous wrapper for chunk scoring."""
        import asyncio
        
        try:
            # Create new event loop for this thread
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._score_chunk_async(query, chunk))
                loop.close()
                return result
            except Exception as e:
                logger.error(f"Error in async execution: {e}")
                return 0.0
        except Exception as e:
            logger.error(f"Error in sync scoring wrapper: {e}")
            return 0.0
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank results using LLM-based relevance scoring."""
        if not results:
            return results
        
        logger.info(f"LLM agent reranking {len(results)} results")
        
        # Limit the number of chunks to score (LLM calls are expensive)
        max_chunks = min(len(results), self.max_chunks_per_batch)
        chunks_to_score = results[:max_chunks]
        remaining_chunks = results[max_chunks:]
        
        scored_results = []
        
        # Score each chunk with the LLM
        for i, result in enumerate(chunks_to_score):
            try:
                llm_score = self._score_chunk_sync(query, result['content'])
                
                new_result = result.copy()
                new_result['llm_relevance_score'] = llm_score
                
                # Combine with original similarity score
                original_score = result.get('similarity_score', 0)
                combined_score = 0.4 * original_score + 0.6 * llm_score
                new_result['combined_score'] = combined_score
                
                scored_results.append(new_result)
                
                logger.debug(f"Chunk {i+1}/{max_chunks}: LLM score={llm_score:.3f}, Combined={combined_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error scoring chunk {i+1}: {e}")
                # Keep original result if scoring fails
                new_result = result.copy()
                new_result['llm_relevance_score'] = 0.0
                new_result['combined_score'] = result.get('similarity_score', 0)
                scored_results.append(new_result)
        
        # Add remaining chunks with their original scores
        for result in remaining_chunks:
            new_result = result.copy()
            new_result['llm_relevance_score'] = None  # Not scored
            new_result['combined_score'] = result.get('similarity_score', 0)
            scored_results.append(new_result)
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"LLM agent reranking completed. Scored {max_chunks} chunks.")
        
        if top_k is not None:
            return scored_results[:top_k]
        return scored_results


def get_reranker(reranker_type: str, **kwargs) -> BaseReranker:
    """Factory function to get reranker instances.
    
    Args:
        reranker_type: Type of reranker ('none', 'bm25', 'keyword_boost', 'cross_encoder', 'multilingual_cross_encoder', 'llm_agent')
        **kwargs: Additional arguments for reranker initialization
        
    Returns:
        Reranker instance
    """
    reranker_type = reranker_type.lower()
    
    if reranker_type in ('none', 'no', 'identity'):
        return NoReranker()
    elif reranker_type == 'bm25':
        return BM25Reranker(**kwargs)
    elif reranker_type in ('keyword', 'keyword_boost'):
        return KeywordBoostReranker(**kwargs)
    elif reranker_type == 'cross_encoder':
        return CrossEncoderReranker(**kwargs)
    elif reranker_type in ('multilingual_cross_encoder', 'multilingual', 'armenian'):
        return MultilingualCrossEncoderReranker(**kwargs)
    elif reranker_type in ('llm_agent', 'llm', 'agent'):
        try:
            return LLMAgentReranker(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM agent reranker: {e}. Falling back to BM25Reranker.")
            return BM25Reranker()
    else:
        logger.warning(f"Unknown reranker type: {reranker_type}. Using NoReranker.")
        return NoReranker()
