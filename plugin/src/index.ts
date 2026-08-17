import type { Context } from '@deepseek-ai/cordis'
import type { PreStepDecision } from '@deepseek-ai/dsh-agent'
import '@deepseek-ai/dsh-tools'
import { appendFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

export const name = 'dsh-adaptive-tool-router'
export const inject = ['tools']

export interface Config {
  /** Shadow mode never restricts tools; it only logs recommendations. */
  shadow?: boolean
  topK?: number
  logPath?: string
}

interface RankedTool {
  name: string
  score: number
}

export function apply(ctx: Context, config: Config = {}): void {
  const shadow = config.shadow ?? true
  const topK = config.topK ?? 5
  if (!shadow) {
    throw new Error('active restriction is intentionally disabled in v0.1; collect shadow data first')
  }
  if (!Number.isInteger(topK) || topK <= 0) {
    throw new Error('topK must be a positive integer')
  }

  ctx.on('agent/pre-step', async (
    { agent, messages, turn, step },
    next,
  ): Promise<PreStepDecision> => {
    const query = messages.map(message => JSON.stringify(message.content)).join('\n')
    const schemas = agent.ctx.tools.schemas()
    const ranked = rankByTokenOverlap(query, schemas).slice(0, topK)

    if (config.logPath) {
      const record = {
        time: new Date().toISOString(),
        mode: 'shadow',
        turn,
        step,
        query,
        candidateCount: schemas.length,
        recommendations: ranked,
      }
      try {
        await mkdir(dirname(config.logPath), { recursive: true })
        await appendFile(config.logPath, `${JSON.stringify(record)}\n`, 'utf8')
      } catch (error) {
        // Observation must never block an agent step.
        console.warn('[dsh-adaptive-tool-router] failed to append shadow log', error)
      }
    }

    return next()
  })
}

function rankByTokenOverlap(
  query: string,
  schemas: ReadonlyArray<{ name: string; description?: string; parameters?: unknown }>,
): RankedTool[] {
  const queryTokens = tokenize(query)
  return schemas
    .map(schema => {
      const document = [
        schema.name,
        schema.description ?? '',
        JSON.stringify(schema.parameters ?? {}),
      ].join(' ')
      const documentTokens = tokenize(document)
      const overlap = [...queryTokens].filter(token => documentTokens.has(token)).length
      const score = queryTokens.size === 0 ? 0 : overlap / queryTokens.size
      return { name: schema.name, score }
    })
    .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name))
}

function tokenize(text: string): Set<string> {
  return new Set(text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [])
}
