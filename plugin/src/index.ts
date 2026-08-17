/**
 * DeepSeek Harness host plugin: rank the current tool catalog before each step.
 *
 * Default Shadow Mode only logs Top-K recommendations. Restriction mode uses
 * `tools.restrict({ allow })` and fail-opens back to the full catalog.
 */
import type { Context } from '@deepseek-ai/cordis'
import { resolveConfig, unique, type RouterConfig } from './config.ts'
import { queryFromMessages } from './query.ts'
import { rankTools, type RankedTool, type ToolSchema } from './rank.ts'

export const name = 'dsh-adaptive-tool-router'
export const inject = ['tools']
export type { RouterConfig }

interface ToolService {
  schemas?: () => ToolSchema[]
  restrict?: (filter: { allow: string[] }) => () => void
}

type PreStepHandler = (
  payload: Record<string, unknown>,
  next: () => unknown,
) => unknown | Promise<unknown>

export function apply(ctx: Context, config: RouterConfig = {}): void {
  const resolved = resolveConfig(config)
  let liftRestriction: (() => void) | undefined

  const runtime = ctx as unknown as {
    on: (event: string, handler: PreStepHandler) => void
    effect?: (factory: () => () => void) => void
  }
  runtime.effect?.(() => () => {
    liftRestriction?.()
    liftRestriction = undefined
  })
  runtime.on('agent/pre-step', async (payload, next) => {
    try {
      const decision = recommend(payload, resolved)
      await persistDecision(resolved, decision)
      if (!resolved.shadow) {
        liftRestriction?.()
        liftRestriction = applyRestriction(payload, decision.allow)
      }
    } catch (error) {
      liftRestriction?.()
      liftRestriction = undefined
      console.warn('[dsh-adaptive-tool-router] fail-open after ranking error', error)
    }

    return next()
  })
}

export function recommend(
  payload: Record<string, unknown>,
  config: ReturnType<typeof resolveConfig>,
): {
  query: string
  candidateCount: number
  recommendations: RankedTool[]
  allow: string[]
} {
  const messages = Array.isArray(payload.messages) ? payload.messages : []
  const query = queryFromMessages(messages)
  const schemas = readSchemas(payload)
  const recommendations = rankTools(query, schemas, {
    limit: config.topK,
    k1: config.k1,
    b: config.b,
  })
  return {
    query,
    candidateCount: schemas.length,
    recommendations,
    allow: unique([
      ...config.alwaysAllow.filter(name => schemas.some(schema => schema.name === name)),
      ...recommendations.map(item => item.name),
    ]),
  }
}

function readSchemas(payload: Record<string, unknown>): ToolSchema[] {
  const tools = readToolService(payload)
  const schemas = tools?.schemas?.() ?? []
  return Array.isArray(schemas) ? schemas : []
}

function applyRestriction(
  payload: Record<string, unknown>,
  allow: string[],
): (() => void) | undefined {
  if (allow.length === 0) {
    return undefined
  }
  const restrict = readToolService(payload)?.restrict
  if (typeof restrict !== 'function') {
    throw new Error('tools.restrict is not available on the agent context')
  }
  return restrict({ allow })
}

function readToolService(payload: Record<string, unknown>): ToolService | undefined {
  const agent = payload.agent as { ctx?: { tools?: ToolService } } | undefined
  return agent?.ctx?.tools
}

async function persistDecision(
  config: ReturnType<typeof resolveConfig>,
  decision: ReturnType<typeof recommend>,
): Promise<void> {
  const names = decision.recommendations.map(item => item.name).join(', ')
  if (config.verbose) {
    const mode = config.shadow ? 'shadow' : 'restrict'
    console.log(
      `[dsh-adaptive-tool-router] ${mode} topK=${config.topK} tools=${decision.candidateCount} recommend=${names || '(none)'}`,
    )
  }
  if (!config.logPath) {
    return
  }

  const { appendFile, mkdir } = await import('node:fs/promises')
  const { dirname } = await import('node:path')
  const record = {
    time: new Date().toISOString(),
    mode: config.shadow ? 'shadow' : 'restrict',
    query: decision.query,
    candidateCount: decision.candidateCount,
    recommendations: decision.recommendations,
    allow: decision.allow,
  }
  try {
    await mkdir(dirname(config.logPath), { recursive: true })
    await appendFile(config.logPath, `${JSON.stringify(record)}\n`, 'utf8')
  } catch (error) {
    console.warn('[dsh-adaptive-tool-router] failed to append decision log', error)
  }
}
