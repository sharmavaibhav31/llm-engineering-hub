"""Tests for reliable_explainer/cli.py."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from reliable_explainer.cli import build_parser, async_main
from reliable_explainer.format import Result
from reliable_explainer.batch import BatchItemResult


def test_cli_argument_parsing():
    parser = build_parser()
    args = parser.parse_args(["What is asyncio?", "-m", "custom-model", "-c", "5", "--no-color"])
    assert args.questions == ["What is asyncio?"]
    assert args.model == "custom-model"
    assert args.concurrency == 5
    assert args.no_color is True
    assert args.batch is False


@pytest.mark.asyncio
async def test_cli_single_question_flow():
    parser = build_parser()
    args = parser.parse_args(["What is a token?"])

    mock_cfg = MagicMock()

    with patch("reliable_explainer.cli.load_config", return_value=mock_cfg):
        with patch("reliable_explainer.cli.handle_single_question", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = Result(
                question="What is a token?",
                status="success",
                attempts=1,
                seconds=0.5,
                answer="A token is a piece of text.",
                tokens=20,
            )
            code = await async_main(args)
            assert code == 0
            mock_handle.assert_called_once_with("What is a token?", config=mock_cfg, use_color=True)


@pytest.mark.asyncio
async def test_cli_batch_questions_flow():
    parser = build_parser()
    args = parser.parse_args(["Q1", "Q2", "-c", "2"])

    mock_cfg = MagicMock()

    with patch("reliable_explainer.cli.load_config", return_value=mock_cfg):
        with patch("reliable_explainer.cli.handle_batch_questions", new_callable=AsyncMock) as mock_handle_batch:
            mock_handle_batch.return_value = [
                Result(question="Q1", status="success", attempts=1, seconds=0.5),
                Result(question="Q2", status="success", attempts=1, seconds=0.5),
            ]
            code = await async_main(args)
            assert code == 0
            mock_handle_batch.assert_called_once()
