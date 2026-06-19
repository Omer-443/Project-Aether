export type Tone = 'emerald' | 'rose' | 'amber' | 'zinc'

export type RiskLabel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'

export type DashboardMetric = {
  label: string
  value: string
  delta: string
  tone: Tone
  description: string
}

export type IngestionPoint = {
  t: number
  claims: number
  denied: number
  latency_ms: number
}

export type DriftAlert = {
  id: string
  time: string
  description: string
  drift_score?: number
  severity: 'high' | 'medium' | 'low'
}

export type DashboardTelemetryResponse = {
  metrics: DashboardMetric[]
  ingestion: IngestionPoint[]
  drifts: DriftAlert[]
  status: Record<string, unknown>
}

export type GapMarker = {
  time: number
  label: string
  days: number
}

export type TrajectoryTelemetryResponse = {
  provider: string
  events: Record<string, unknown>[]
  trajectory: Array<{
    time: number
    z: number
    friction: number
    dim_0: number
    dim_1: number
    dim_2: number
    dim_3: number
    [k: string]: unknown
  }>
  gaps: GapMarker[]
  denial_probability: number
  risk_label: RiskLabel
  stats: Record<string, unknown>
}

export type LogsRow = {
  id: string
  timestamp: string
  cpt_opaque_id: string
  continuous_time: number
  latent_state: number
  friction_score: number
  status: 'APPROVED' | 'DENIED' | 'PENDING' | 'APPEALED'
}

export type LogsTelemetryResponse = {
  rows: LogsRow[]
  summary: Record<string, number>
}

export type ClaimEvent = {
  claim_date: string
  allowed_amount: number
  billed_amount: number
  drg_weight: number
  length_of_stay: number
  icd_chapter: number
  procedure_count: number
  prior_denial_rate: number
  payer_score: number
}

export type SampleDataResponse = {
  events: ClaimEvent[]
  provider: string
  n_events: number
  date_range: {
    start: string
    end: string
  }
}

export type ShockTestRequest = {
  events: ClaimEvent[]
  shock_magnitude: number
  shock_type: 'step' | 'impulse' | 'ramp'
}

export type ShockTestResponse = {
  baseline_prob: number
  shocked_prob: number
  delta_prob: number
  adaptation_score: number
  shock_magnitude: number
  shock_type: 'step' | 'impulse' | 'ramp'
  baseline_trajectory: Array<Record<string, number>>
  shocked_trajectory: Array<Record<string, number>>
  eval_times: number[]
  interpretation: string
}

const DEFAULT_BASE_URL = 'http://localhost:8000'

function getBaseUrl() {
  const envBaseUrl = (process as any).env?.NEXT_PUBLIC_AETHER_API_BASE_URL
  return String(envBaseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, '')
}




async function apiGet<T>(path: string, query?: Record<string, string | number>) {

  const baseUrl = getBaseUrl()
  const url = new URL(baseUrl + path)
  if (query) {
    for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v))
  }

  // eslint-disable-next-line no-console
  console.log('[aether-client] GET', url.toString())
  const res = await fetch(url.toString(), { credentials: 'include' })
  // eslint-disable-next-line no-console
  console.log('[aether-client] GET response', res.status, url.toString())

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    // eslint-disable-next-line no-console
    console.error('[aether-client] GET failed body:', text)
    throw new Error(`Aether API GET ${path} failed: ${res.status} ${text}`)
  }
  return (await res.json()) as T

}

async function apiPost<T>(path: string, body: unknown) {
  const baseUrl = getBaseUrl()
  const res = await fetch(baseUrl + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Aether API POST ${path} failed: ${res.status} ${text}`)
  }
  return (await res.json()) as T
}

export async function getDashboardTelemetry(): Promise<DashboardTelemetryResponse> {
  return apiGet<DashboardTelemetryResponse>('/api/v1/telemetry/dashboard')
}

export async function getTrajectoryTelemetry(nEvents: number): Promise<TrajectoryTelemetryResponse> {
  return apiGet<TrajectoryTelemetryResponse>('/api/v1/telemetry/trajectory', { n_events: nEvents })
}

export async function getLogsTelemetry(limit: number): Promise<LogsTelemetryResponse> {
  return apiGet<LogsTelemetryResponse>('/api/v1/telemetry/logs', { limit })
}

export async function getSampleData(nEvents: number): Promise<SampleDataResponse> {
  return apiGet<SampleDataResponse>('/api/v1/sample-data', { n_events: nEvents })
}

export async function runShockTest(
  events: ClaimEvent[],
  shockMagnitude: number,
  shockType: ShockTestRequest['shock_type'] = 'step'
): Promise<ShockTestResponse> {
  const shock_magnitude = Math.max(0.01, Math.min(2, shockMagnitude))
  return apiPost<ShockTestResponse>('/api/v1/shock-test', { events, shock_magnitude, shock_type: shockType })
}

