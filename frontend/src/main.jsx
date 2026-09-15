import React, { StrictMode, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const HERO_BACKGROUND = 'https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=2400&q=85'
const FOOTER_BACKGROUND = 'https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=2400&q=85'
const PAGE_BACKGROUND = 'https://images.unsplash.com/photo-1533283725824-a62a971989ac?auto=format&fit=crop&fm=jpg&q=82&w=2400'
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const pitchCards = [
  {
    title: 'Multi-Modal, Cross-Source Registration',
    subtitle: 'Aligning Different Sensors on the Same Moon',
    text: 'Registers lunar imagery captured by different sensors, resolutions, viewing conditions, and illumination—establishing reliable correspondences across heterogeneous observations.',
    statement: 'DIFFERENT SENSORS. SAME TERRAIN. ONE ALIGNMENT.',
    image: 'https://images.unsplash.com/photo-1447433589675-4aaa569f3e05?auto=format&fit=crop&w=1200&q=85',
  },
  {
    title: 'Physics-Guided Tie-Point Matching',
    subtitle: 'From Image Features to Lunar Geometry',
    text: 'Combines image correspondences with lunar terrain and viewing geometry to improve the reliability of tie points, especially where appearance alone is difficult.',
    statement: 'WHEN APPEARANCE CHANGES, GEOMETRY GUIDES.',
    image: 'https://images.unsplash.com/photo-1528722828814-77b9b83aafb2?auto=format&fit=crop&w=1200&q=85',
  },
  {
    title: 'Adaptive Multi-Scale Processing',
    subtitle: 'One Pipeline. Many Image Scales.',
    text: 'Automatically works at an efficient scale for correspondence search while preserving original-resolution coordinates for accurate refinement and registration.',
    statement: 'ADAPTIVE SCALE. CONSISTENT RESULTS.',
    image: 'https://images.unsplash.com/photo-1502134249126-9f3755a50d78?auto=format&fit=crop&w=1200&q=85',
  },
  {
    title: 'High-Precision Alignment & Validation',
    subtitle: 'Every Tie Point Must Earn Its Place',
    text: 'Uses confidence filtering, uniform spatial selection, RANSAC, sub-pixel refinement, and quantitative error analysis to produce reliable final correspondences.',
    statement: 'FROM PIXELS TO MEASURABLE ACCURACY.',
    image: 'https://images.unsplash.com/photo-1464802686167-b939a6910659?auto=format&fit=crop&w=1200&q=85',
  },
  {
    title: 'Multi-Mission Lunar Analysis',
    subtitle: 'A Common Reference for Changing Observations',
    text: 'Creates a consistent spatial relationship between observations from different lunar missions, enabling comparison and analysis across datasets.',
    statement: 'ONE MOON. MANY MISSIONS. CONNECTED DATA.',
    image: 'https://images.unsplash.com/photo-1538370965046-79c0d6907d47?auto=format&fit=crop&w=1200&q=85',
  },
  {
    title: 'Smarter Lunar Exploration',
    subtitle: 'Better Alignment. Better Lunar Intelligence.',
    text: 'Reliable registration supports scientific analysis, terrain interpretation, landing-site studies, change analysis, and integration of future lunar observations.',
    statement: 'BETTER DATA. CLEARER DECISIONS.',
    image: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1200&q=85',
  },
]

const stages = [
  'Preparing Images',
  'Finding Correspondences',
  'Filtering Points',
  'Estimating Transformation',
  'Refining Tie Points',
  'Registering Image',
  'Validating Results',
]

const metricLabels = [
  'Source Keypoints',
  'Reference Keypoints',
  'Candidate Matches',
  'Final Tie Points',
  'RANSAC Outliers',
  'RANSAC Inliers',
  'Inlier Ratio',
  'Mean Reprojection Error',
  'Median Reprojection Error',
  'Maximum Reprojection Error',
  'RMSE',
  'Registration Status',
  'Processing Time',
]

function isUnavailable(value) {
  return value === undefined || value === null || value === '' || value === 'unavailable' || value === 'None' || value === 'null'
}

function metricValue(label, result) {
  const metrics = result?.metrics || {}
  const values = {
    'Source Keypoints': metrics.source_keypoints,
    'Reference Keypoints': metrics.reference_keypoints,
    'Candidate Matches': metrics.candidate_matches,
    'Final Tie Points': result?.tie_points?.length,
    'RANSAC Inliers': metrics.ransac_inliers,
    'RANSAC Outliers': metrics.ransac_outliers,
    'Inlier Ratio': metrics.inlier_ratio,
    'Mean Reprojection Error': metrics.mean_reprojection_error_px,
    'Median Reprojection Error': metrics.median_reprojection_error_px,
    'Maximum Reprojection Error': metrics.maximum_reprojection_error_px,
    'RMSE': metrics.rmse_reprojection_error_px,
    'Registration Status': result?.status,
    'Processing Time': result?.processing_time_seconds ?? metrics.total_pipeline_runtime_seconds,
  }
  const value = values[label]
  if (isUnavailable(value)) return '—'
  if (label === 'Registration Status') {
    return value === 'success' ? 'Successful' : value === 'insufficient_matches' ? 'Insufficient matches' : value === 'failed' ? 'Failed' : String(value)
  }
  if (label === 'Inlier Ratio') return `${(Number(value) * 100).toFixed(2)}%`
  if (label === 'Processing Time') return `${Number(value).toFixed(2)} s`
  if (label.includes('Error') || label.includes('RMSE') || label.includes('Residual')) return `${Number(value).toFixed(2)} px`
  return typeof value === 'number' ? (Number.isInteger(value) ? String(value) : value.toFixed(2)) : String(value)
}

function formatCell(value) {
  if (isUnavailable(value)) return 'Not available'
  if (typeof value === 'number') return value.toFixed(2)
  return String(value)
}

function Icon({ name, size = 18, stroke = 'currentColor' }) {
  const paths = {
    layers: <><path d="m3 7 9-4 9 4-9 4-9-4Z" /><path d="m3 12 9 4 9-4" /><path d="m3 17 9 4 9-4" /></>,
    upload: <><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M4 16v3h16v-3" /></>,
    play: <path d="m8 5 10 7-10 7V5Z" fill="currentColor" stroke="none" />,
    arrow: <><path d="M4 12h15" /><path d="m14 7 5 5-5 5" /></>,
    crosshair: <><circle cx="12" cy="12" r="7" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3" /></>,
    image: <><rect x="3" y="4" width="18" height="16" rx="1" /><circle cx="8" cy="9" r="1.5" /><path d="m4 17 5-5 3 3 2-2 6 5" /></>,
    chart: <><path d="M4 19V5M4 19h16" /><path d="m7 15 3-4 3 2 5-7" /></>,
    table: <><rect x="3" y="4" width="18" height="16" rx="1" /><path d="M3 9h18M3 14h18M8 9v11M15 9v11" /></>,
    download: <><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M4 20h16" /></>,
    search: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 5 5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    menu: <><path d="M4 7h16M4 12h16M4 17h16" /></>,
  }
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>
}

function SectionHeading({ eyebrow, title, description, icon }) {
  return <div className="section-heading">
    <div className="section-heading-main">
      {icon && <span className="heading-icon"><Icon name={icon} size={19} /></span>}
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h2>{title}</h2>
        {description && <p className="section-description">{description}</p>}
      </div>
    </div>
  </div>
}

function fileSize(bytes) {
  if (!bytes) return '—'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function UploadCard({ label, value, onChange }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const acceptFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    const image = new Image()
    image.onload = () => onChange({ file, url, width: image.naturalWidth, height: image.naturalHeight })
    image.src = url
  }

  return <article className={`upload-card ${dragging ? 'is-dragging' : ''}`}>
    <div className="upload-card-header"><span className="card-icon"><Icon name="layers" size={18} /></span><h3>{label}</h3></div>
    <div className="upload-card-body">
      <div className="drop-zone"
        onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files?.[0]) }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex="0"
        onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click() }}
      >
        {value ? <img src={value.url} alt={`${label} preview`} className="upload-preview" /> : <>
          <span className="upload-symbol"><Icon name="upload" size={24} /></span>
          <strong>Drag and drop an image here</strong>
          <span className="drop-or">or</span>
          <button type="button" className="browse-button" onClick={(event) => { event.stopPropagation(); inputRef.current?.click() }}>Browse Files</button>
          <small>Supports PNG, JPG, TIFF, etc.</small>
        </>}
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(event) => acceptFile(event.target.files?.[0])} />
      </div>
      <div className="file-meta">
        {value ? <>
          <p className="file-name" title={value.file.name}>{value.file.name}</p>
          <p>{value.width} × {value.height} pixels</p>
          <p>{fileSize(value.file.size)}</p>
        </> : <p className="empty-file-meta">No image selected</p>}
      </div>
    </div>
  </article>
}

function ProcessingSection({ active, result, error }) {
  const status = active ? 'Working...' : error ? 'Registration failed' : result?.status === 'success' ? 'Complete' : result?.status === 'insufficient_matches' ? 'Insufficient matches' : 'Ready when you are'
  return <section className="content-section processing-section" id="processing">
    <div className="processing-topline">
      <div className="section-heading-main"><span className="heading-icon processing-icon"><Icon name="crosshair" size={19} /></span><div><h2>Processing lunar images...</h2><p className="section-description">FastAPI → run_pipeline.py → generated registration outputs</p></div></div>
      <span className={`processing-status ${active ? 'active' : ''} ${error ? 'error' : ''}`}>{status}</span>
    </div>
    <div className={`timeline ${active ? 'is-processing' : ''}`} aria-label="Processing stages">
      <div className="timeline-line" />
      {stages.map((item, index) => <div className={`timeline-step ${active && index === 0 ? 'active' : ''} ${!active && !result && index === 0 ? 'ready' : ''}`} key={item}>
        <span className="timeline-dot">{!active && result?.status === 'success' ? <Icon name="check" size={11} /> : ''}</span>
        <span>{item}</span>
      </div>)}
    </div>
  </section>
}

function EmptyVisual({ title, icon = 'image' }) {
  return <div className="empty-visual"><span className="empty-visual-icon"><Icon name={icon} size={26} /></span><strong>{title}</strong><span>Results will appear after registration</span></div>
}

function CorrespondenceSection({ result }) {
  const image = result?.outputs?.candidate_matches || result?.outputs?.final_tie_points
  return <section className="content-section" id="correspondences">
    <SectionHeading title="Corresponding Tie Points" description="Matched keypoints between source and reference images" icon="crosshair" />
    <div className="correspondence-empty">{image ? <img className="backend-visual" src={image} alt="Generated candidate correspondences" /> : <EmptyVisual title="Results will appear after registration" icon="crosshair" />}</div>
    <div className="visual-legend empty-legend"><span><i className="legend-dot inlier" /> Inlier Match</span><span><i className="legend-dot outlier" /> Outlier Match</span><span><i className="legend-dot point" /> Tie Point</span></div>
  </section>
}

function RegistrationResults({ result }) {
  const panels = [
    ['Registered Source', result?.outputs?.registered_source],
    ['Overlay', result?.outputs?.overlay],
    ['Difference Image', result?.outputs?.difference],
  ]
  return <section className="content-section" id="results">
    <SectionHeading title="Registration Result" description="Aligned source image, overlay, and difference visualization" icon="image" />
    <div className="result-stage"><div className="result-grid">
      {panels.map(([title, image]) => <article className="result-card" key={title}>{image ? <img className="backend-visual result-image" src={image} alt={title} /> : <EmptyVisual title="No result available" />}<span className="result-label">{title}</span></article>)}
    </div></div>
  </section>
}

function MetricsSection({ result }) {
  return <section className="content-section" id="validation">
    <SectionHeading title="Quantitative Validation" description="Key metrics and evaluation results" icon="chart" />
    <div className="metrics-grid">
      {metricLabels.map((label) => <div className="metric-card" key={label}><span>{label}</span><strong className={label === 'Registration Status' ? 'status-empty' : ''}>{metricValue(label, result)}</strong></div>)}
    </div>
  </section>
}

function TiePointsTable({ result }) {
  const rows = result?.tie_points || []
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(1)
  const pageSize = 5
  useEffect(() => setPage(1), [result?.run_id, query])
  const filteredRows = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return rows
    return rows.filter((row) => Object.values(row).some((value) => String(value ?? '').toLowerCase().includes(normalized)))
  }, [rows, query])
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize))
  const pageRows = filteredRows.slice((page - 1) * pageSize, page * pageSize)
  return <section className="content-section" id="tie-points">
    <div className="table-heading-row"><SectionHeading title="Final Tie Points" description="Corresponding points in original image coordinates" icon="table" /><div className="search-box"><Icon name="search" size={15} /><input placeholder="Search tie points..." value={query} disabled={!rows.length} onChange={(event) => setQuery(event.target.value)} /></div></div>
    <div className="table-wrap"><table><thead><tr>{['#', 'Source X', 'Source Y', 'Reference X', 'Reference Y', 'Confidence', 'Match Distance', 'Reproj. Error', 'Inlier'].map((head) => <th key={head}>{head}</th>)}</tr></thead><tbody>{pageRows.length ? pageRows.map((row, index) => <tr key={`${row.source_x}-${row.reference_x}-${index}`}><td>{(page - 1) * pageSize + index + 1}</td><td>{formatCell(row.source_x)}</td><td>{formatCell(row.source_y)}</td><td>{formatCell(row.reference_x)}</td><td>{formatCell(row.reference_y)}</td><td>{formatCell(row.confidence)}</td><td>{formatCell(row.match_distance)}</td><td>{formatCell(row.reprojection_error)}</td><td>{row.inlier === true || row.inlier === 'true' ? '✓' : '—'}</td></tr>) : <tr><td colSpan="9"><div className="table-empty"><Icon name="table" size={24} /><strong>No tie points available</strong><span>{result ? 'The backend returned no accepted correspondences.' : 'Upload images and complete registration to populate this table.'}</span></div></td></tr>}</tbody></table></div>
    <div className="table-footer"><span>{filteredRows.length ? `Showing ${(page - 1) * pageSize + 1}–${Math.min(page * pageSize, filteredRows.length)} of ${filteredRows.length} tie points` : 'Showing 0 tie points'}</span><div className="pagination"><button disabled={page <= 1} onClick={() => setPage((current) => Math.max(1, current - 1))}>‹</button>{filteredRows.length ? <button className="current" disabled>{page} / {pageCount}</button> : <button className="current" disabled>1</button>}<button disabled={page >= pageCount} onClick={() => setPage((current) => Math.min(pageCount, current + 1))}>›</button></div></div>
  </section>
}

function DownloadAction({ label, url }) {
  const [downloading, setDownloading] = useState(false)
  const filename = url ? decodeURIComponent(url.split('/').pop() || `${label.toLowerCase().replaceAll(' ', '_')}.dat`) : ''

  const downloadFile = async () => {
    if (!url || downloading) return
    setDownloading(true)
    try {
      const response = await fetch(url)
      if (!response.ok) throw new Error(`Download failed with status ${response.status}`)
      const blob = await response.blob()
      const objectUrl = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = filename
      anchor.style.display = 'none'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.setTimeout(() => window.URL.revokeObjectURL(objectUrl), 1000)
    } catch (downloadError) {
      console.error(downloadError)
    } finally {
      setDownloading(false)
    }
  }

  return <button className="download-button" disabled={!url || downloading} onClick={downloadFile} title={url ? `Download ${filename}` : 'Available after registration'}><Icon name="download" size={17} /><span>{downloading ? 'Preparing download...' : label}</span></button>
}

function DownloadSection({ result }) {
  const downloads = [
    ['Download Registered Image', result?.outputs?.registered_source],
    ['Download Overlay', result?.outputs?.overlay],
    ['Download Difference', result?.outputs?.difference],
    ['Download Tie Points CSV', result?.outputs?.tie_points_csv],
    ['Download Metrics JSON', result?.outputs?.metrics_json],
    ['Download Report', result?.outputs?.report],
  ]
  return <section className="content-section download-section" id="downloads">
    <SectionHeading title="Download Results" description="Generated outputs for further analysis" icon="download" />
    <div className="download-grid">{downloads.map(([label, url]) => <DownloadAction key={label} label={label} url={url} />)}</div>
  </section>
}

function AboutSection() {
  return <section className="about-section" id="about">
    <div className="about-intro">
      <p className="eyebrow">THE PROJECT</p>
      <h2>From lunar imagery<br /><em>to useful insight.</em></h2>
      <p>Visionary is building a careful image-registration workflow for lunar observations—connecting modern matching with transparent geometry, validation, and repeatable research outputs.</p>
    </div>
    <div className="pitch-grid">
      {pitchCards.map((card, index) => <article className="pitch-card" key={card.title}>
        <div className="pitch-card-image"><img src={card.image} alt="" loading="lazy" /><span>{String(index + 1).padStart(2, '0')}</span></div>
        <div className="pitch-card-body"><p className="pitch-tag">{String(index + 1).padStart(2, '0')} <i /></p><h3>{card.title}</h3><h4>{card.subtitle}</h4><p>{card.text}</p><p className="pitch-statement">{card.statement}</p></div>
      </article>)}
    </div>
    <div className="team-contact" id="contact">
      <div><p className="eyebrow">CONTACT</p><h2>Built by Team Visionary.</h2><p>For collaboration, demonstrations, and project enquiries, reach the team directly.</p></div>
      <div className="contact-details"><a href="tel:9655920225">+91 96559 20225</a><a href="mailto:r.sanjairsk@gmail.com">r.sanjairsk@gmail.com</a></div>
    </div>
  </section>
}

function App() {
  const [source, setSource] = useState(null)
  const [reference, setReference] = useState(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const requestControllerRef = useRef(null)

  const handleSource = (value) => { setSource(value) }
  const handleReference = (value) => { setReference(value) }

  const resetProcess = () => {
    requestControllerRef.current?.abort()
    if (source?.url) window.URL.revokeObjectURL(source.url)
    if (reference?.url) window.URL.revokeObjectURL(reference.url)
    requestControllerRef.current = null
    setSource(null)
    setReference(null)
    setRunning(false)
    setResult(null)
    setError(null)
    document.querySelector('#upload')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const handleRegister = async () => {
    if (!source || !reference || running) return
    setRunning(true)
    setResult(null)
    setError(null)
    document.querySelector('#processing')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    const controller = new AbortController()
    requestControllerRef.current = controller
    try {
      const form = new FormData()
      form.append('source', source.file, source.file.name)
      form.append('reference', reference.file, reference.file.name)
      const response = await fetch(`${API_BASE}/api/register`, { method: 'POST', body: form, signal: controller.signal })
      const payload = await response.json().catch(() => ({}))
      setResult(payload)
      if (!response.ok || payload.status === 'failed') {
        setError(payload.error || 'The registration pipeline failed.')
      } else if (payload.status === 'insufficient_matches') {
        setError('The pipeline completed but found insufficient matches. Diagnostics are still shown below.')
      } else {
        document.querySelector('#results')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    } catch (requestError) {
      if (requestError.name !== 'AbortError') setError(`Could not reach the FastAPI backend: ${requestError.message}`)
    } finally {
      if (requestControllerRef.current === controller) requestControllerRef.current = null
      setRunning(false)
    }
  }

  const runDisabled = !source || !reference || running
  const uploadedState = useMemo(() => Boolean(source || reference), [source, reference])

  return <div className="app-shell">
    <header className="site-nav">
      <a className="brand" href="#home" aria-label="Lunar Image Registration home"><span className="brand-mark">◐</span><span><strong>LUNAR</strong><small>IMAGE REGISTRATION</small></span></a>
      <button className="mobile-menu" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle navigation"><Icon name="menu" size={20} /></button>
      <nav className={menuOpen ? 'nav-links open' : 'nav-links'}>{['Home', 'About', 'Contact'].map((link, index) => <a className={index === 0 ? 'active' : ''} href={`#${link.toLowerCase() === 'home' ? 'home' : link.toLowerCase()}`} key={link}>{link}</a>)}</nav>
      <div className="nav-utility"><span className="nav-search"><Icon name="search" size={15} /></span><span>ALIGN</span><span>IMAGE</span><span>OPS</span></div>
    </header>

    <main style={{ '--page-image': `url("${PAGE_BACKGROUND}")` }}>
      <section className="hero" id="home" style={{ '--hero-image': `url("${HERO_BACKGROUND}")` }}>
        <div className="hero-image" />
        <div className="hero-content page-width"><p className="hero-kicker">IMAGES<br />ALIGN<br />WORLDS</p><h1><span>Lunar</span><span>Image <em>Registration</em></span></h1><p className="hero-copy">Establishing accurate image correspondences and registration between lunar observations to enable better analysis and understanding of the Moon.</p><div className="hero-rule" /><p className="hero-caption">FROM LUNAR IMAGERY<br />TO NEW INSIGHTS</p></div>
        <div className="hero-side-note">EXPLORE<br />THE SURFACE<br />BEYOND</div>
      </section>

      <section className="content-section upload-section" id="upload">
        <div className="upload-heading-row"><SectionHeading eyebrow="WORKSPACE" title="Upload Images" description="Select a source image and a reference image to establish corresponding points and register the imagery." /><button className="reset-button" type="button" onClick={resetProcess}><span aria-hidden="true">↺</span> New Upload / Reset</button></div>
        <div className="upload-grid"><UploadCard label="Source Image" value={source} onChange={handleSource} /><UploadCard label="Reference Image" value={reference} onChange={handleReference} /></div>
        <div className="run-row"><button className="run-button" disabled={runDisabled} onClick={handleRegister}><Icon name="play" size={16} /> <span>{running ? 'Processing...' : 'Run Registration'}</span> <Icon name="arrow" size={18} /></button>{!uploadedState && <p className="run-hint">Upload both images to enable registration.</p>}</div>
        {error && <p className="api-error" role="alert">{error}</p>}
      </section>

      <ProcessingSection active={running} result={result} error={error} />
      <CorrespondenceSection result={result} />
      <RegistrationResults result={result} />
      <MetricsSection result={result} />
      <TiePointsTable result={result} />
      <DownloadSection result={result} />
      <AboutSection />
    </main>

    <footer className="site-footer" style={{ '--footer-image': `url("${FOOTER_BACKGROUND}")` }}><div className="footer-image" /><div className="footer-content"><p className="footer-side">THE MOON<br />CONNECTS US<br />ALL</p><div><p className="footer-title">Different Perspectives.</p><p className="footer-title accent">A Clearer Moon.</p></div><span className="footer-side footer-link">PLANETARY<br />IMAGING SYSTEMS <Icon name="crosshair" size={13} /></span></div></footer>
  </div>
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
