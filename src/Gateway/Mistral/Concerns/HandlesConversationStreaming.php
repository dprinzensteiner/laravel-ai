<?php

namespace Laravel\Ai\Gateway\Mistral\Concerns;

use Generator;
use Laravel\Ai\Gateway\StepResponse;
use Laravel\Ai\Providers\Provider;
use Laravel\Ai\Responses\Data\FinishReason;
use Laravel\Ai\Responses\Data\Meta;
use Laravel\Ai\Responses\Data\ToolCall;
use Laravel\Ai\Responses\Data\Usage;
use Laravel\Ai\Streaming\Events\Error;
use Laravel\Ai\Streaming\Events\StreamEvent;
use Laravel\Ai\Streaming\Events\StreamStart;
use Laravel\Ai\Streaming\Events\TextDelta;
use Laravel\Ai\Streaming\Events\TextEnd;
use Laravel\Ai\Streaming\Events\TextStart;
use Laravel\Ai\Streaming\Events\ToolCall as ToolCallEvent;

trait HandlesConversationStreaming
{
    /** @return Generator<int, StreamEvent, mixed, StepResponse|null> */
    protected function processConversationStream(
        string $invocationId,
        Provider $provider,
        string $model,
        $streamBody,
    ): Generator {
        $messageId = $this->generateEventId();
        $streamStartEmitted = false;
        $textStartEmitted = false;
        $currentText = '';
        $pendingToolCalls = [];
        $toolCalls = [];
        $usage = null;
        $responseModel = $model;

        foreach ($this->parseServerSentEvents($streamBody) as $data) {
            $type = $data['type'] ?? null;

            if ($type === 'conversation.response.error') {
                yield (new Error(
                    $this->generateEventId(),
                    (string) ($data['code'] ?? 'unknown_error'),
                    $data['message'] ?? 'Unknown error',
                    false,
                    time(),
                ))->withInvocationId($invocationId);

                return null;
            }

            if ($type === 'conversation.response.done') {
                $usage = $this->extractUsage($data);

                continue;
            }

            if (! in_array($type, ['message.output.delta', 'function.call.delta'], true)) {
                continue;
            }

            if (! $streamStartEmitted) {
                $streamStartEmitted = true;
                $responseModel = $data['model'] ?? $model;

                yield (new StreamStart(
                    $this->generateEventId(),
                    $provider->name(),
                    $responseModel,
                    time(),
                ))->withInvocationId($invocationId);
            }

            if ($type === 'function.call.delta') {
                $pendingToolCalls[$data['tool_call_id']] ??= ['name' => $data['name'] ?? '', 'arguments' => ''];
                $pendingToolCalls[$data['tool_call_id']]['arguments'] .= $data['arguments'] ?? '';

                continue;
            }

            $content = $this->extractConversationText($data['content'] ?? '');

            if ($content === '') {
                continue;
            }

            if (! $textStartEmitted) {
                $textStartEmitted = true;

                yield (new TextStart(
                    $this->generateEventId(),
                    $messageId,
                    time(),
                ))->withInvocationId($invocationId);
            }

            $currentText .= $content;

            yield (new TextDelta(
                $this->generateEventId(),
                $messageId,
                $content,
                time(),
            ))->withInvocationId($invocationId);
        }

        if ($textStartEmitted) {
            yield (new TextEnd(
                $this->generateEventId(),
                $messageId,
                time(),
            ))->withInvocationId($invocationId);
        }

        foreach ($pendingToolCalls as $id => $pending) {
            $toolCall = new ToolCall(
                $id,
                $pending['name'],
                json_decode($pending['arguments'] ?: '{}', true) ?? [],
                $id,
            );

            $toolCalls[] = $toolCall;

            yield (new ToolCallEvent(
                $this->generateEventId(),
                $toolCall,
                time(),
            ))->withInvocationId($invocationId);
        }

        return new StepResponse(
            text: $currentText,
            toolCalls: $toolCalls,
            finishReason: filled($toolCalls) ? FinishReason::ToolCalls : FinishReason::Stop,
            usage: $usage ?? new Usage(0, 0),
            meta: new Meta($provider->name(), $responseModel),
        );
    }
}
