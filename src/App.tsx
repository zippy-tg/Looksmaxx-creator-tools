import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL } from '@ffmpeg/util';
import { toCanvas } from 'html-to-image';
import { ArrowLeft, ChevronRight, ImagePlus, ScanFace } from 'lucide-react';
import './index.css';

type Screen =
  | 'home'
  | 'face-analysis'
  | 'height-posture-analysis'
  | 'category-intro'
  | 'minimal-rating-potential'
  | 'detailed-breakdown'
  | 'detailed-breakdown-2'
  | 'ascend'
  | 'height-posture-detailed';
type CategoryScreen =
  | 'minimal-rating-potential'
  | 'detailed-breakdown'
  | 'detailed-breakdown-2'
  | 'ascend'
  | 'height-posture-detailed';

interface ScorecardState {
  image: string | null;
  sourceImage: string | null;
  imageOffsetX: number;
  imageOffsetY: number;
  imageZoom: number;
  minimalScore: number;
  minimalTier: string;
  potentialScore: number;
  potentialTier: string;
  ratingGradientStart: string;
  ratingGradientEnd: string;
  potentialGradientStart: string;
  potentialGradientEnd: string;
  ratingBarStart: string;
  ratingBarEnd: string;
  potentialBarStart: string;
  potentialBarEnd: string;
  avatarBorderStart: string;
  avatarBorderEnd: string;
  meshColor: string;
  meshOpacity: number;
}

interface DetailedBreakdownState {
  frontImage: string | null;
  sideImage: string | null;
  frontSourceImage: string | null;
  sideSourceImage: string | null;
  frontImageOffsetX: number;
  frontImageOffsetY: number;
  sideImageOffsetX: number;
  sideImageOffsetY: number;
  frontImageZoom: number;
  sideImageZoom: number;
  swapHeaderWithAvatars: boolean;
  overallScore: number;
  harmonyScore: number;
  angularityScore: number;
  dimorphismScore: number;
  featuresScore: number;
  symmetryScore: number;
  proportionsScore: number;
  accentStart: string;
  accentEnd: string;
  scoreColor: string;
  landmarkColor: string;
  landmarkOpacity: number;
  landmarkDotSize: number;
  landmarkLineThickness: number;
  badgeBorderStart: string;
  badgeBorderEnd: string;
}

interface AscendState {
  image: string | null;
  sourceImage: string | null;
  imageOffsetX: number;
  imageOffsetY: number;
  imageZoom: number;
  pslScore: number;
  pslTier: string;
  potentialScore: number;
  potentialTier: string;
  improvementScore: number;
  accentStart: string;
  accentEnd: string;
  fontColor: string;
  avatarBorderStart: string;
  avatarBorderEnd: string;
  meshColor: string;
  meshOpacity: number;
}

const INITIAL_CARD: ScorecardState = {
  image: null,
  sourceImage: null,
  imageOffsetX: 0,
  imageOffsetY: 0,
  imageZoom: 1,
  minimalScore: 5.2,
  minimalTier: 'HTN',
  potentialScore: 7.4,
  potentialTier: 'CHAD',
  ratingGradientStart: '#2a080a',
  ratingGradientEnd: '#09090b',
  potentialGradientStart: '#32090b',
  potentialGradientEnd: '#09090b',
  ratingBarStart: '#ff3b30',
  ratingBarEnd: '#ff8a65',
  potentialBarStart: '#ff4035',
  potentialBarEnd: '#ff8e72',
  avatarBorderStart: '#4a6cff',
  avatarBorderEnd: '#7da2ff',
  meshColor: '#ffffff',
  meshOpacity: 0.42,
};

const TIER_OPTIONS = ['SUB 5', 'LTN', 'MTN', 'HTN', 'CHAD LITE', 'CHAD', 'ADAM', 'LTB', 'MTB', 'HTB', 'STACY'] as const;

const INITIAL_DETAILED_CARD: DetailedBreakdownState = {
  frontImage: null,
  sideImage: null,
  frontSourceImage: null,
  sideSourceImage: null,
  frontImageOffsetX: 0,
  frontImageOffsetY: 0,
  sideImageOffsetX: 0,
  sideImageOffsetY: 0,
  frontImageZoom: 1,
  sideImageZoom: 1,
  swapHeaderWithAvatars: false,
  overallScore: 6.4,
  harmonyScore: 7.2,
  angularityScore: 5.4,
  dimorphismScore: 6.0,
  featuresScore: 5.8,
  symmetryScore: 6.6,
  proportionsScore: 6.2,
  accentStart: '#c7d99f',
  accentEnd: '#f1dfb7',
  scoreColor: '#d8e7b2',
  landmarkColor: '#ff5a52',
  landmarkOpacity: 0.88,
  landmarkDotSize: 3.2,
  landmarkLineThickness: 1.6,
  badgeBorderStart: '#9cb5ff',
  badgeBorderEnd: '#496ef9',
};

const INITIAL_HEIGHT_POSTURE_CARD: DetailedBreakdownState = {
  frontImage: null,
  sideImage: null,
  frontSourceImage: null,
  sideSourceImage: null,
  frontImageOffsetX: 0,
  frontImageOffsetY: 0,
  sideImageOffsetX: 0,
  sideImageOffsetY: 0,
  frontImageZoom: 1,
  sideImageZoom: 1,
  swapHeaderWithAvatars: false,
  overallScore: 6.8,
  harmonyScore: 7.4,
  angularityScore: 6.9,
  dimorphismScore: 7.1,
  featuresScore: 6.6,
  symmetryScore: 6.8,
  proportionsScore: 7.3,
  accentStart: '#a8d9b8',
  accentEnd: '#d8eed4',
  scoreColor: '#d8eed4',
  landmarkColor: '#ff5a52',
  landmarkOpacity: 0.88,
  landmarkDotSize: 3.2,
  landmarkLineThickness: 1.6,
  badgeBorderStart: '#9cb5ff',
  badgeBorderEnd: '#496ef9',
};

const INITIAL_ASCEND_CARD: AscendState = {
  image: null,
  sourceImage: null,
  imageOffsetX: 0,
  imageOffsetY: 0,
  imageZoom: 1,
  pslScore: 6.8,
  pslTier: 'HTN',
  potentialScore: 8.1,
  potentialTier: 'CHAD',
  improvementScore: 7.6,
  accentStart: '#4a6cff',
  accentEnd: '#8ec5ff',
  fontColor: '#ffffff',
  avatarBorderStart: '#7da2ff',
  avatarBorderEnd: '#3f6bff',
  meshColor: '#ffffff',
  meshOpacity: 0.42,
};

const FACE_MESH_MODEL = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const FACE_MESH_EDGES = faceLandmarksDetection.util.getAdjacentPairs(FACE_MESH_MODEL);
const BODY_POSE_MODEL = poseDetection.SupportedModels.MoveNet;
const BODY_CONNECTIONS: Array<[number, number]> = [
  [0, 5],
  [0, 6],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

let faceMeshDetectorPromise: Promise<Awaited<ReturnType<typeof faceLandmarksDetection.createDetector>>> | null = null;
let poseDetectorPromise: Promise<Awaited<ReturnType<typeof poseDetection.createDetector>>> | null = null;
let ffmpegPromise: Promise<FFmpeg> | null = null;

type FaceStatus = 'idle' | 'detecting' | 'detected' | 'no-face' | 'error';
type MeshPoint = { x: number; y: number };
type AnimationDurationMs = 1500 | 3000;
type DetectionMode = 'face' | 'body';
type DetailedMetrics = {
  eyeSpacingPercent: number;
  jawWidthPercent: number;
  midfacePercent: number;
  symmetryPercent: number;
};
type DetailOverlayStyle = {
  color: string;
  opacity: number;
  dotSize: number;
  lineThickness: number;
};

const EXPORT_LAYOUT_SIZE = 460;
const EXPORT_VIDEO_SIZE = 1080;
const EXPORT_IMAGE_SIZE = 2160;
const EXPORT_PORTRAIT_LAYOUT_WIDTH = 360;
const EXPORT_PORTRAIT_LAYOUT_HEIGHT = 640;
const EXPORT_PORTRAIT_VIDEO_WIDTH = 1080;
const EXPORT_PORTRAIT_VIDEO_HEIGHT = 1920;
const EXPORT_PORTRAIT_IMAGE_WIDTH = 2160;
const EXPORT_PORTRAIT_IMAGE_HEIGHT = 3840;

async function getFaceMeshDetector() {
  if (!faceMeshDetectorPromise) {
    faceMeshDetectorPromise = (async () => {
      await tf.setBackend('webgl');
      await tf.ready();

      return faceLandmarksDetection.createDetector(FACE_MESH_MODEL, {
        runtime: 'tfjs',
        refineLandmarks: true,
        maxFaces: 1,
      });
    })();
  }

  return faceMeshDetectorPromise;
}

async function getPoseDetector() {
  if (!poseDetectorPromise) {
    poseDetectorPromise = (async () => {
      await tf.setBackend('webgl');
      await tf.ready();

      return poseDetection.createDetector(BODY_POSE_MODEL, {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
      });
    })();
  }

  return poseDetectorPromise;
}

async function getFFmpeg() {
  if (!ffmpegPromise) {
    const ffmpeg = new FFmpeg();
    ffmpegPromise = (async () => {
      await ffmpeg.load({
        coreURL: await toBlobURL('/ffmpeg-core.js', 'text/javascript'),
        wasmURL: await toBlobURL('/ffmpeg-core.wasm', 'application/wasm'),
      });

      return ffmpeg;
    })();
  }

  return ffmpegPromise;
}

function App() {
  const [screen, setScreen] = useState<Screen>('home');
  const [pendingCategory, setPendingCategory] = useState<CategoryScreen | null>(null);
  const [card, setCard] = useState<ScorecardState>(INITIAL_CARD);
  const [detailedCard, setDetailedCard] = useState<DetailedBreakdownState>(INITIAL_DETAILED_CARD);
  const [detailedCard2, setDetailedCard2] = useState<DetailedBreakdownState>(INITIAL_DETAILED_CARD);
  const [heightPostureCard, setHeightPostureCard] = useState<DetailedBreakdownState>(INITIAL_HEIGHT_POSTURE_CARD);
  const [ascendCard, setAscendCard] = useState<AscendState>(INITIAL_ASCEND_CARD);
  const fileRef = useRef<HTMLInputElement>(null);
  const detailedFrontFileRef = useRef<HTMLInputElement>(null);
  const detailedSideFileRef = useRef<HTMLInputElement>(null);
  const detailed2FrontFileRef = useRef<HTMLInputElement>(null);
  const detailed2SideFileRef = useRef<HTMLInputElement>(null);
  const ascendFileRef = useRef<HTMLInputElement>(null);

  const readImageInput = (
    event: ChangeEvent<HTMLInputElement>,
    onLoad: (imageData: string) => void,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const imageData = reader.result;

      if (typeof imageData === 'string') {
        onLoad(imageData);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleImage = (event: ChangeEvent<HTMLInputElement>) => {
    readImageInput(event, (imageData) => {
      setCard((current) => ({
        ...current,
        image: imageData,
        sourceImage: imageData,
        imageOffsetX: 0,
        imageOffsetY: 0,
        imageZoom: 1,
      }));
    });
  };

  const handleDetailedFrontImage = (event: ChangeEvent<HTMLInputElement>) => {
    readImageInput(event, (imageData) => {
      setDetailedCard((current) => ({
        ...current,
        frontImage: imageData,
        frontSourceImage: imageData,
        frontImageOffsetX: 0,
        frontImageOffsetY: 0,
        frontImageZoom: 1,
      }));
    });
  };

  const handleHeightPostureFrontImage = (event: ChangeEvent<HTMLInputElement>) => {
    readImageInput(event, (imageData) => {
      setHeightPostureCard((current) => ({
        ...current,
        frontImage: imageData,
        frontSourceImage: imageData,
        frontImageOffsetX: 0,
        frontImageOffsetY: 0,
        frontImageZoom: 1,
      }));
    });
  };

  const handleAscendImage = (event: ChangeEvent<HTMLInputElement>) => {
    readImageInput(event, (imageData) => {
      setAscendCard((current) => ({
        ...current,
        image: imageData,
        sourceImage: imageData,
        imageOffsetX: 0,
        imageOffsetY: 0,
        imageZoom: 1,
      }));
    });
  };

  const handleDetailedSideImage = (event: ChangeEvent<HTMLInputElement>) => {
    readImageInput(event, (imageData) => {
      setDetailedCard((current) => ({
        ...current,
        sideImage: imageData,
        sideSourceImage: imageData,
        sideImageOffsetX: 0,
        sideImageOffsetY: 0,
        sideImageZoom: 1,
      }));
    });
  };

  const handleDetailed2FrontImage = (event: ChangeEvent<HTMLInputElement>) => {
    readImageInput(event, (imageData) => {
      setDetailedCard2((current) => ({
        ...current,
        frontImage: imageData,
        frontSourceImage: imageData,
        frontImageOffsetX: 0,
        frontImageOffsetY: 0,
        frontImageZoom: 1,
      }));
    });
  };

  const handleDetailed2SideImage = (event: ChangeEvent<HTMLInputElement>) => {
    readImageInput(event, (imageData) => {
      setDetailedCard2((current) => ({
        ...current,
        sideImage: imageData,
        sideSourceImage: imageData,
        sideImageOffsetX: 0,
        sideImageOffsetY: 0,
        sideImageZoom: 1,
      }));
    });
  };

  const setDetailedField = <K extends keyof DetailedBreakdownState>(
    field: K,
    value: DetailedBreakdownState[K],
  ) => {
    setDetailedCard((current) => ({ ...current, [field]: value }));
  };

  const setDetailed2Field = <K extends keyof DetailedBreakdownState>(
    field: K,
    value: DetailedBreakdownState[K],
  ) => {
    setDetailedCard2((current) => ({ ...current, [field]: value }));
  };

  const setHeightPostureField = <K extends keyof DetailedBreakdownState>(
    field: K,
    value: DetailedBreakdownState[K],
  ) => {
    setHeightPostureCard((current) => ({ ...current, [field]: value }));
  };

  const setField = <K extends keyof ScorecardState>(field: K, value: ScorecardState[K]) => {
    setCard((current) => ({ ...current, [field]: value }));
  };

  const setAscendField = <K extends keyof AscendState>(field: K, value: AscendState[K]) => {
    setAscendCard((current) => ({ ...current, [field]: value }));
  };

  const openCategory = (nextScreen: CategoryScreen) => {
    setPendingCategory(nextScreen);
    setScreen('category-intro');
  };

  const categoryParentScreen = pendingCategory === 'height-posture-detailed' ? 'height-posture-analysis' : 'face-analysis';

  return (
    <div className="shell">
      {screen === 'home' && (
        <HomeScreen
          onOpenFaceAnalysis={() => setScreen('face-analysis')}
          onOpenHeightPosture={() => setScreen('height-posture-analysis')}
        />
      )}

      {screen === 'face-analysis' && (
        <FaceAnalysisScreen
          onBack={() => setScreen('home')}
          onOpenMinimal={() => openCategory('minimal-rating-potential')}
          onOpenDetailed={() => openCategory('detailed-breakdown')}
          onOpenDetailed2={() => openCategory('detailed-breakdown-2')}
          onOpenAscend={() => openCategory('ascend')}
        />
      )}

      {screen === 'category-intro' && pendingCategory && (
        <CategoryIntroScreen
          category={pendingCategory}
          onBack={() => setScreen(categoryParentScreen)}
          onContinue={() => setScreen(pendingCategory)}
        />
      )}

      {screen === 'minimal-rating-potential' && (
        <MinimalEditorScreen
          card={card}
          fileRef={fileRef}
          onBack={() => setScreen('face-analysis')}
          onOpenPicker={() => fileRef.current?.click()}
          onImageChange={handleImage}
          onChange={setField}
        />
      )}

      {screen === 'detailed-breakdown' && (
        <DetailedBreakdownEditorScreen
          card={detailedCard}
          frontFileRef={detailedFrontFileRef}
          sideFileRef={detailedSideFileRef}
          onBack={() => setScreen('face-analysis')}
          onOpenFrontPicker={() => detailedFrontFileRef.current?.click()}
          onOpenSidePicker={() => detailedSideFileRef.current?.click()}
          onFrontImageChange={handleDetailedFrontImage}
          onSideImageChange={handleDetailedSideImage}
          onChange={setDetailedField}
        />
      )}

      {screen === 'detailed-breakdown-2' && (
        <DetailedBreakdownEditorScreen
          card={detailedCard2}
          frontFileRef={detailed2FrontFileRef}
          sideFileRef={detailed2SideFileRef}
          onBack={() => setScreen('face-analysis')}
          onOpenFrontPicker={() => detailed2FrontFileRef.current?.click()}
          onOpenSidePicker={() => detailed2SideFileRef.current?.click()}
          onFrontImageChange={handleDetailed2FrontImage}
          onSideImageChange={handleDetailed2SideImage}
          onChange={setDetailed2Field}
          title="Detailed Breakdown 2.0"
          description="Build the 9:16 dual-angle breakdown card for creator edits."
          templateLabel="Detailed 2.0"
          enableSideAvatar
          exportBaseName="omnimaxx-detailed-breakdown-2"
        />
      )}

      {screen === 'ascend' && (
        <AscendEditorScreen
          card={ascendCard}
          fileRef={ascendFileRef}
          onBack={() => setScreen('face-analysis')}
          onOpenPicker={() => ascendFileRef.current?.click()}
          onImageChange={handleAscendImage}
          onChange={setAscendField}
        />
      )}

      {screen === 'height-posture-analysis' && (
        <HeightPostureAnalysisScreen
          onBack={() => setScreen('home')}
          onOpenDetailed={() => openCategory('height-posture-detailed')}
        />
      )}

      {screen === 'height-posture-detailed' && (
        <DetailedBreakdownEditorScreen
          card={heightPostureCard}
          frontFileRef={detailedFrontFileRef}
          onBack={() => setScreen('height-posture-analysis')}
          onOpenFrontPicker={() => detailedFrontFileRef.current?.click()}
          onFrontImageChange={handleHeightPostureFrontImage}
          onChange={setHeightPostureField}
          title="Detailed Height/Posture Analysis"
          description="Build the vertical 9:16 height and posture card for creator edits."
          templateLabel="Height/Posture"
          scoreLabels={['Overall', 'Height', 'Posture', 'Frame', 'Presence', 'Balance', 'Proportions']}
          scanLabelsDetected={['Height read', 'Posture read', 'Frame read', 'Balance read', 'Compiling results']}
          detectionMode="body"
        />
      )}
    </div>
  );
}

function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function HomeScreen({
  onOpenFaceAnalysis,
  onOpenHeightPosture,
}: {
  onOpenFaceAnalysis: () => void;
  onOpenHeightPosture: () => void;
}) {
  return (
    <div className="home-shell">
      <header className="home-header">
        <h1 className="home-title">LooksMaxx Creator Tools</h1>
      </header>

      <main className="home-main">
        <button className="tool-card" onClick={onOpenFaceAnalysis}>
          <div className="tool-card__left">
            <div className="tool-card__icon">
              <ScanFace size={24} />
            </div>
            <div>
              <h2>Face Analysis</h2>
              <p>Create scorecards for creator videos</p>
            </div>
          </div>

          <ChevronRight size={20} />
        </button>

        <button className="tool-card" onClick={onOpenHeightPosture}>
          <div className="tool-card__left">
            <div className="tool-card__icon">
              <ScanFace size={24} />
            </div>
            <div>
              <h2>Height/Posture Analysis</h2>
              <p>Create body analysis layouts for creator videos</p>
            </div>
          </div>

          <ChevronRight size={20} />
        </button>
      </main>
    </div>
  );
}

function FaceAnalysisScreen({
  onBack,
  onOpenMinimal,
  onOpenDetailed,
  onOpenDetailed2,
  onOpenAscend,
}: {
  onBack: () => void;
  onOpenMinimal: () => void;
  onOpenDetailed: () => void;
  onOpenDetailed2: () => void;
  onOpenAscend: () => void;
}) {
  return (
    <div className="page page--simple">
      <button className="back-button" onClick={onBack}>
        <ArrowLeft size={16} />
        <span>Back</span>
      </button>

      <div className="section-copy">
        <h1>Face Analysis</h1>
        <p>Choose a scorecard format.</p>
      </div>

      <div className="category-stack">
        <button className="category-card category-card--active" onClick={onOpenMinimal}>
          <div className="category-card__copy">
            <h2>Minimal Rating & Potential</h2>
            <p>Clean two-box layout for creator edits</p>
            <div className="category-card__badges">
              <span className="category-card__meta">Photo + video</span>
              <span className="category-card__tag">Classic</span>
            </div>
          </div>
          <ChevronRight size={18} />
        </button>

        <button className="category-card category-card--active" onClick={onOpenDetailed}>
          <div className="category-card__copy">
            <h2>Detailed Breakdown</h2>
            <p>Portrait analysis layout for vertical creator edits</p>
            <div className="category-card__badges">
              <span className="category-card__meta">Photo + video</span>
              <span className="category-card__tag">Cinematic</span>
            </div>
          </div>
          <ChevronRight size={18} />
        </button>

        <button className="category-card category-card--active" onClick={onOpenDetailed2}>
          <div className="category-card__copy">
            <h2>Detailed Breakdown 2.0</h2>
            <p>Same detailed layout with front and side avatars</p>
            <div className="category-card__badges">
              <span className="category-card__meta">Photo + video</span>
              <span className="category-card__tag category-card__tag--warn">Buggy - trying to make it work</span>
            </div>
          </div>
          <ChevronRight size={18} />
        </button>

        <button className="category-card category-card--active" onClick={onOpenAscend}>
          <div className="category-card__copy">
            <h2>ASCEND</h2>
            <p>9:16 glow-up projection layout</p>
            <span className="category-card__meta">Photo + video</span>
          </div>
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}

function CategoryIntroScreen({
  category,
  onBack,
  onContinue,
}: {
  category: CategoryScreen;
  onBack: () => void;
  onContinue: () => void;
}) {
  const categoryLabel =
    category === 'minimal-rating-potential'
      ? 'Minimal Rating & Potential'
      : category === 'detailed-breakdown'
        ? 'Detailed Breakdown'
        : category === 'detailed-breakdown-2'
          ? 'Detailed Breakdown 2.0'
        : category === 'ascend'
          ? 'ASCEND'
          : 'Detailed Height/Posture Analysis';

  return (
    <div className="page page--simple">
      <button className="back-button" onClick={onBack}>
        <ArrowLeft size={16} />
        <span>Back</span>
      </button>

      <div className="section-copy">
        <h1>{categoryLabel}</h1>
        <p>Before you start, one quick creator note.</p>
      </div>

      <div className="category-intro-card">
        <span className="category-intro-card__pill">Photo + video</span>
        <h2>Please use face mesh</h2>
        <p>It helps us represent the app feature more clearly in creator content.</p>
        <p>Using the mesh makes the scorecards and promo exports feel closer to the real OmniMaxx experience.</p>

        <div className="category-intro-card__actions">
          <button className="back-button" type="button" onClick={onContinue}>
            <span>Continue</span>
          </button>
        </div>
      </div>
    </div>
  );
}

function HeightPostureAnalysisScreen({
  onBack,
  onOpenDetailed,
}: {
  onBack: () => void;
  onOpenDetailed: () => void;
}) {
  return (
    <div className="page page--simple">
      <button className="back-button" onClick={onBack}>
        <ArrowLeft size={16} />
        <span>Back</span>
      </button>

      <div className="section-copy">
        <h1>Height/Posture Analysis</h1>
        <p>Choose a scorecard format.</p>
      </div>

      <div className="category-stack">
        <button className="category-card category-card--active" onClick={onOpenDetailed}>
          <div className="category-card__copy">
            <h2>Detailed Height/Posture Analysis</h2>
            <p>Portrait-style posture and height layout for creator edits</p>
            <span className="category-card__meta">Photo + video</span>
          </div>
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}

function MinimalEditorScreen({
  card,
  fileRef,
  onBack,
  onOpenPicker,
  onImageChange,
  onChange,
}: {
  card: ScorecardState;
  fileRef: React.RefObject<HTMLInputElement | null>;
  onBack: () => void;
  onOpenPicker: () => void;
  onImageChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onChange: <K extends keyof ScorecardState>(field: K, value: ScorecardState[K]) => void;
}) {
  const [animationNonce, setAnimationNonce] = useState(0);
  const [animationDurationMs, setAnimationDurationMs] = useState<AnimationDurationMs>(3000);
  const [animationProgress, setAnimationProgress] = useState(0);
  const [draftImageOffsetX, setDraftImageOffsetX] = useState(card.imageOffsetX);
  const [draftImageOffsetY, setDraftImageOffsetY] = useState(card.imageOffsetY);
  const [draftImageZoom, setDraftImageZoom] = useState(card.imageZoom);
  const [draftPreviewImage, setDraftPreviewImage] = useState<string | null>(card.image);
  const [exportFrameProgress, setExportFrameProgress] = useState<number | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const [faceStatus, setFaceStatus] = useState<FaceStatus>('idle');
  const [meshPoints, setMeshPoints] = useState<MeshPoint[]>([]);
  const [avatarRenderTick, setAvatarRenderTick] = useState(0);
  const avatarImageRef = useRef<HTMLImageElement>(null);
  const meshCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportAvatarImageRef = useRef<HTMLImageElement>(null);
  const exportMeshCanvasRef = useRef<HTMLCanvasElement>(null);
  const previewCardRef = useRef<HTMLDivElement>(null);
  const exportPreviewCardRef = useRef<HTMLDivElement>(null);
  const effectiveFaceStatus: FaceStatus = card.image ? faceStatus : 'idle';
  const isCropDirty =
    draftImageOffsetX !== card.imageOffsetX ||
    draftImageOffsetY !== card.imageOffsetY ||
    draftImageZoom !== card.imageZoom;
  const previewImage = draftPreviewImage ?? card.image;
  const effectiveAnimationProgress = exportFrameProgress ?? animationProgress;
  const topbarReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0, 0.18));
  const avatarReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.12, 0.35));
  const ratingReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.23, 0.43));
  const potentialReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.27, 0.47));
  const displayedMinimalScore = card.minimalScore * easedProgress(windowedProgress(effectiveAnimationProgress, 0.42, 0.93));
  const displayedPotentialScore =
    card.potentialScore * easedProgress(windowedProgress(effectiveAnimationProgress, 0.48, 0.96));
  const ratingBarProgress = card.minimalScore * 10 * easedProgress(windowedProgress(effectiveAnimationProgress, 0.34, 0.78));
  const potentialBarProgress =
    card.potentialScore * 10 * easedProgress(windowedProgress(effectiveAnimationProgress, 0.4, 0.84));
  const meshDrawProgress = easedProgress(windowedProgress(effectiveAnimationProgress, 0.16, 0.78));
  const animationSeed = [
    animationNonce,
    card.image ?? 'no-image',
    card.minimalScore,
    card.minimalTier,
    card.potentialScore,
    card.potentialTier,
    card.ratingGradientStart,
    card.ratingGradientEnd,
    card.potentialGradientStart,
    card.potentialGradientEnd,
    card.ratingBarStart,
    card.ratingBarEnd,
    card.potentialBarStart,
    card.potentialBarEnd,
    card.avatarBorderStart,
    card.avatarBorderEnd,
    animationDurationMs,
  ].join('|');

  useEffect(() => {
    let frameId = 0;
    const start = performance.now();

    const tick = (now: number) => {
      const next = Math.min((now - start) / animationDurationMs, 1);
      setAnimationProgress(next);

      if (next < 1) {
        frameId = requestAnimationFrame(tick);
      }
    };

    frameId = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(frameId);
  }, [animationSeed, animationDurationMs]);

  useEffect(() => {
    let cancelled = false;
    let detectFrameId = 0;

    if (!card.image) {
      return;
    }

    detectFrameId = requestAnimationFrame(() => {
      if (!cancelled) {
        setFaceStatus('detecting');
        setMeshPoints([]);
      }
    });

    const image = new Image();
    image.crossOrigin = 'anonymous';

    image.onload = async () => {
      try {
        const detector = await getFaceMeshDetector();
        const faces = await detector.estimateFaces(image, {
          flipHorizontal: false,
          staticImageMode: true,
        });

        if (cancelled) {
          return;
        }

        if (faces.length === 0) {
          setFaceStatus('no-face');
          setMeshPoints([]);
          return;
        }

        setFaceStatus('detected');
        setMeshPoints(faces[0].keypoints.map((point) => ({ x: point.x, y: point.y })));
      } catch {
        if (!cancelled) {
          setFaceStatus('error');
          setMeshPoints([]);
        }
      }
    };

    image.onerror = () => {
      if (!cancelled) {
        setFaceStatus('error');
        setMeshPoints([]);
      }
    };

    image.src = card.image;

    return () => {
      cancelled = true;
      cancelAnimationFrame(detectFrameId);
    };
  }, [card.image]);

  useEffect(() => {
    setDraftImageOffsetX(card.imageOffsetX);
    setDraftImageOffsetY(card.imageOffsetY);
    setDraftImageZoom(card.imageZoom);
  }, [card.imageOffsetX, card.imageOffsetY, card.imageZoom, card.sourceImage]);

  useEffect(() => {
    let cancelled = false;

    if (!card.sourceImage) {
      setDraftPreviewImage(card.image);
      return;
    }

    createCroppedAvatarImage({
      source: card.sourceImage,
      offsetX: draftImageOffsetX,
      offsetY: draftImageOffsetY,
      zoom: draftImageZoom,
      size: 1400,
    })
      .then((croppedImage) => {
        if (!cancelled) {
          setDraftPreviewImage(croppedImage);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setDraftPreviewImage(card.image);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [card.image, card.sourceImage, draftImageOffsetX, draftImageOffsetY, draftImageZoom]);

  useEffect(() => {
    drawFaceMesh({
      canvas: meshCanvasRef.current,
      image: avatarImageRef.current,
      status: effectiveFaceStatus,
      meshPoints,
      meshDrawProgress,
      meshColor: card.meshColor,
      meshOpacity: card.meshOpacity,
    });

    drawFaceMesh({
      canvas: exportMeshCanvasRef.current,
      image: exportAvatarImageRef.current,
      status: effectiveFaceStatus,
      meshPoints,
      meshDrawProgress,
      meshColor: card.meshColor,
      meshOpacity: card.meshOpacity,
    });
  }, [
    avatarRenderTick,
    card.meshColor,
    card.meshOpacity,
    effectiveFaceStatus,
    meshDrawProgress,
    meshPoints,
  ]);

  const handleRenderMesh = async () => {
    if (!card.sourceImage) {
      return;
    }

    const croppedImage = await createCroppedAvatarImage({
      source: card.sourceImage,
      offsetX: draftImageOffsetX,
      offsetY: draftImageOffsetY,
      zoom: draftImageZoom,
      size: 1400,
    });

    onChange('imageOffsetX', draftImageOffsetX);
    onChange('imageOffsetY', draftImageOffsetY);
    onChange('imageZoom', draftImageZoom);
    onChange('image', croppedImage);
    setDraftPreviewImage(croppedImage);
    setAnimationNonce((current) => current + 1);
  };

  const handleExportVideo = async () => {
    if (!previewCardRef.current || isExporting) {
      return;
    }

    setIsExporting(true);
    setExportMessage(null);
    setExportProgress(0);

    try {
      setExportMessage('Loading MP4 exporter...');
      setExportProgress(5);
      const ffmpeg = await getFFmpeg();
      const ffmpegLogs: string[] = [];
      const logListener = ({ message }: { message: string }) => {
        ffmpegLogs.push(message);
        if (ffmpegLogs.length > 20) {
          ffmpegLogs.shift();
        }
      };
      const progressListener = ({ progress }: { progress: number }) => {
        const normalized = Math.max(0, Math.min(1, progress));
        setExportProgress(70 + Math.round(normalized * 25));
        setExportMessage(`Encoding MP4... ${Math.round(normalized * 100)}%`);
      };

      ffmpeg.on('log', logListener);
      ffmpeg.on('progress', progressListener);
      setExportProgress(12);

      const sourceNode = exportPreviewCardRef.current;
      if (!sourceNode) {
        throw new Error('Could not prepare the export preview.');
      }

      await waitForExportNodeReady(sourceNode);
      setAnimationNonce((current) => current + 1);
      await wait(80);
      setExportFrameProgress(0);
      await waitForAnimationFrame();
      await waitForAnimationFrame();
      const width = EXPORT_LAYOUT_SIZE;
      const height = EXPORT_LAYOUT_SIZE;
      const outputSize = EXPORT_VIDEO_SIZE;
      const fps = 30;
      const totalFrames = Math.max(2, Math.round((animationDurationMs / 1000) * fps));
      const exportId = `${Date.now()}`;
      const inputPattern = `frame-${exportId}-%03d.png`;
      const outputName = `omnimaxx-scorecard-${exportId}.mp4`;

      setExportMessage('Rendering frames...');
      setExportProgress(18);

      for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        const frameProgress = frameIndex / (totalFrames - 1);
        setExportFrameProgress(frameProgress);
        await waitForAnimationFrame();
        drawFaceMesh({
          canvas: exportMeshCanvasRef.current,
          image: exportAvatarImageRef.current,
          status: effectiveFaceStatus,
          meshPoints,
          meshDrawProgress: easedProgress(windowedProgress(frameProgress, 0.16, 0.78)),
          meshColor: card.meshColor,
          meshOpacity: card.meshOpacity,
        });
        await waitForAnimationFrame();

        const frame = await toCanvas(sourceNode, {
          cacheBust: true,
          width,
          height,
          canvasWidth: outputSize,
          canvasHeight: outputSize,
          pixelRatio: 1,
          backgroundColor: '#000000',
          skipAutoScale: true,
          style: {
            width: `${width}px`,
            height: `${height}px`,
          },
          filter: (node) => !(node instanceof HTMLElement && node.dataset.exportIgnore === 'true'),
        });
        const blob = await canvasToBlob(frame);
        const fileName = `frame-${exportId}-${frameIndex.toString().padStart(3, '0')}.png`;
        await ffmpeg.writeFile(fileName, new Uint8Array(await blob.arrayBuffer()));
        const renderProgress = (frameIndex + 1) / totalFrames;
        setExportProgress(18 + Math.round(renderProgress * 47));
        setExportMessage(`Rendering frames... ${Math.round(renderProgress * 100)}%`);
      }

      setExportFrameProgress(1);
      await waitForAnimationFrame();

      setExportMessage('Encoding MP4...');
      setExportProgress(70);
      let exitCode = await ffmpeg.exec([
        '-framerate',
        String(fps),
        '-i',
        inputPattern,
        '-c:v',
        'libx264',
        '-preset',
        'slower',
        '-crf',
        '15',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        outputName,
      ]);

      if (exitCode !== 0) {
        exitCode = await ffmpeg.exec([
          '-framerate',
          String(fps),
          '-i',
          inputPattern,
          '-c:v',
          'mpeg4',
          '-q:v',
          '2',
          '-pix_fmt',
          'yuv420p',
          '-movflags',
          '+faststart',
          outputName,
        ]);
      }

      if (exitCode !== 0) {
        const tail = ffmpegLogs.slice(-3).join(' | ').trim();
        throw new Error(tail ? `FFmpeg failed: ${tail}` : `FFmpeg failed with code ${exitCode}.`);
      }

      const data = await ffmpeg.readFile(outputName);
      const bytes = data instanceof Uint8Array ? data : new TextEncoder().encode(String(data));
      const safeBytes = new Uint8Array(bytes.length);
      safeBytes.set(bytes);
      const blob = new Blob([safeBytes], { type: 'video/mp4' });
      downloadBlob(blob, outputName);
      setExportMessage('MP4 exported.');
      setExportProgress(100);

      for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        const fileName = `frame-${exportId}-${frameIndex.toString().padStart(3, '0')}.png`;
        await ffmpeg.deleteFile(fileName);
      }

      await ffmpeg.deleteFile(outputName);
      ffmpeg.off('log', logListener);
      ffmpeg.off('progress', progressListener);
    } catch (error) {
      setExportMessage(describeExportError(error));
      setExportProgress(null);
    } finally {
      setExportFrameProgress(null);
      setIsExporting(false);
    }
  };

  const handleExportImage = async () => {
    if (!previewCardRef.current || isExporting) {
      return;
    }

    setIsExporting(true);
    setExportMessage('Rendering 4K PNG...');
    setExportProgress(0);

    try {
      setExportFrameProgress(1);
      await wait(80);
      await waitForAnimationFrame();

      const sourceNode = exportPreviewCardRef.current;
      if (!sourceNode) {
        throw new Error('Could not prepare the export preview.');
      }
      const width = EXPORT_LAYOUT_SIZE;
      const height = EXPORT_LAYOUT_SIZE;
      const captureSize = EXPORT_VIDEO_SIZE;
      const outputSize = EXPORT_IMAGE_SIZE;
      const outputName = `omnimaxx-scorecard-${Date.now()}.png`;

      setExportProgress(25);
      await waitForExportNodeReady(sourceNode);
      drawFaceMesh({
        canvas: exportMeshCanvasRef.current,
        image: exportAvatarImageRef.current,
        status: effectiveFaceStatus,
        meshPoints,
        meshDrawProgress: 1,
        meshColor: card.meshColor,
        meshOpacity: card.meshOpacity,
      });
      await waitForAnimationFrame();

      const frame = await renderExportCanvas({
        node: sourceNode,
        width,
        height,
        canvasWidth: captureSize,
        canvasHeight: captureSize,
      });

      setExportProgress(75);
      const upscaledFrame = upscaleCanvas(frame, outputSize, outputSize);
      const blob = await canvasToBlob(upscaledFrame);
      downloadBlob(blob, outputName);
      setExportMessage('4K PNG exported.');
      setExportProgress(100);
    } catch (error) {
      setExportMessage(describeExportError(error));
      setExportProgress(null);
    } finally {
      setExportFrameProgress(null);
      setIsExporting(false);
    }
  };

  return (
    <div className="editor-shell">
      <button className="back-button" onClick={onBack}>
        <ArrowLeft size={16} />
        <span>Back</span>
      </button>

      <div className="editor-header">
        <h1>Profile Editor</h1>
        <p>Customize the values for your scorecard mockup</p>
      </div>

      <div className="editor-layout">
        <aside className="editor-left">
          <div className="template-pill">Minimal</div>

          <div className="compact-grid compact-grid--top">
            <Field label="Minimal Rating">
              <select
                className="control"
                value={card.minimalTier}
                onChange={(event) => onChange('minimalTier', event.target.value)}
              >
                {TIER_OPTIONS.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </Field>

            <Field label="Avatar Image">
              <button className="control control--upload" onClick={onOpenPicker}>
                <span>{card.image ? 'Change image' : 'Choose file'}</span>
              </button>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                hidden
                onChange={onImageChange}
              />
              <span className={`face-status face-status--${effectiveFaceStatus}`}>{getFaceStatusLabel(effectiveFaceStatus)}</span>
            </Field>

            <Field label="Animation Length">
              <select
                className="control"
                value={animationDurationMs}
                onChange={(event) => setAnimationDurationMs(Number(event.target.value) as AnimationDurationMs)}
              >
                <option value={1500}>1.5 sec</option>
                <option value={3000}>3 sec</option>
              </select>
            </Field>

          </div>

          <div className="editor-section">
            <h2>Minimal Template</h2>

            <div className="compact-grid compact-grid--triple">
              <Field label="Minimal Score">
                <input
                  className="control"
                  type="number"
                  min={0}
                  max={10}
                  step={0.1}
                  value={card.minimalScore}
                  onChange={(event) => onChange('minimalScore', Number(event.target.value))}
                />
              </Field>

              <Field label="Potential Rating">
                <select
                  className="control"
                  value={card.potentialTier}
                  onChange={(event) => onChange('potentialTier', event.target.value)}
                >
                  {TIER_OPTIONS.map((option) => (
                    <option key={option}>{option}</option>
                  ))}
                </select>
              </Field>

              <Field label="Potential Score">
                <input
                  className="control"
                  type="number"
                  min={0}
                  max={10}
                  step={0.1}
                  value={card.potentialScore}
                  onChange={(event) => onChange('potentialScore', Number(event.target.value))}
                />
              </Field>
            </div>
          </div>

          <div className="editor-section">
            <h2>Colors</h2>

            <div className="compact-grid compact-grid--double">
              <ColorField
                label="Rating Gradient Start"
                value={card.ratingGradientStart}
                onChange={(value) => onChange('ratingGradientStart', value)}
              />
              <ColorField
                label="Rating Gradient End"
                value={card.ratingGradientEnd}
                onChange={(value) => onChange('ratingGradientEnd', value)}
              />
              <ColorField
                label="Potential Gradient Start"
                value={card.potentialGradientStart}
                onChange={(value) => onChange('potentialGradientStart', value)}
              />
              <ColorField
                label="Potential Gradient End"
                value={card.potentialGradientEnd}
                onChange={(value) => onChange('potentialGradientEnd', value)}
              />
              <ColorField
                label="Rating Bar Start"
                value={card.ratingBarStart}
                onChange={(value) => onChange('ratingBarStart', value)}
              />
              <ColorField
                label="Rating Bar End"
                value={card.ratingBarEnd}
                onChange={(value) => onChange('ratingBarEnd', value)}
              />
              <ColorField
                label="Potential Bar Start"
                value={card.potentialBarStart}
                onChange={(value) => onChange('potentialBarStart', value)}
              />
              <ColorField
                label="Potential Bar End"
                value={card.potentialBarEnd}
                onChange={(value) => onChange('potentialBarEnd', value)}
              />
              <ColorField
                label="Avatar Border Start"
                value={card.avatarBorderStart}
                onChange={(value) => onChange('avatarBorderStart', value)}
              />
              <ColorField
                label="Avatar Border End"
                value={card.avatarBorderEnd}
                onChange={(value) => onChange('avatarBorderEnd', value)}
              />
              <ColorField
                label="Mesh Color"
                value={card.meshColor}
                onChange={(value) => onChange('meshColor', value)}
              />
            </div>
          </div>

          <div className="editor-section">
            <h2>Mesh</h2>

            <div className="compact-grid compact-grid--double">
              <TuneInput
                label="Mesh Opacity"
                value={card.meshOpacity}
                min={0.1}
                max={1}
                step={0.01}
                onChange={(value) => onChange('meshOpacity', value)}
              />
              <div className="field">
                <span className="field__label">Image Position</span>
                <div className="control control--static">Use the crop sliders below</div>
              </div>
            </div>
          </div>

          {card.sourceImage && (
            <div className="editor-section">
              <div className="avatar-crop-card">
                <div className="avatar-crop-card__header">
                  <h2>Avatar crop</h2>
                  <button
                    type="button"
                    className="control control--ghost control--ghost-small"
                    onClick={() => {
                      setDraftImageZoom(1);
                      setDraftImageOffsetX(0);
                      setDraftImageOffsetY(0);
                    }}
                  >
                    Reset
                  </button>
                </div>

                <div className="avatar-crop-card__preview">
                  <div
                    className="avatar-crop-card__ring"
                    style={{
                      background: `linear-gradient(135deg, ${card.avatarBorderStart} 0%, ${card.avatarBorderEnd} 100%)`,
                      boxShadow: `0 0 24px ${card.avatarBorderStart}33, 0 0 36px ${card.avatarBorderEnd}1f`,
                    }}
                  >
                    <div className="avatar-crop-card__avatar">
                      <img
                        src={previewImage ?? card.sourceImage}
                        alt="Avatar crop preview"
                        className="avatar-crop-card__image"
                        draggable={false}
                      />
                    </div>
                  </div>
                </div>

                <div className="avatar-crop-card__controls">
                  <CropSlider
                    label="Zoom"
                    valueLabel={`${draftImageZoom.toFixed(2)}x`}
                    value={draftImageZoom}
                    min={1}
                    max={3}
                    step={0.01}
                    onChange={setDraftImageZoom}
                  />
                  <CropSlider
                    label="Horizontal"
                    valueLabel={`${Math.round(draftImageOffsetX)}px`}
                    value={draftImageOffsetX}
                    min={-150}
                    max={150}
                    step={1}
                    onChange={setDraftImageOffsetX}
                  />
                  <CropSlider
                    label="Vertical"
                    valueLabel={`${Math.round(draftImageOffsetY)}px`}
                    value={draftImageOffsetY}
                    min={-150}
                    max={150}
                    step={1}
                    onChange={setDraftImageOffsetY}
                  />
                  <button
                    type="button"
                    className="preview-export"
                    onClick={handleRenderMesh}
                  >
                    Render mesh
                  </button>
                </div>
              </div>
            </div>
          )}
        </aside>

        <section className="editor-right">
          <div className="preview-stage">
            <PreviewCard
              innerRef={previewCardRef}
              card={card}
              previewImage={previewImage}
              suppressMesh={isCropDirty}
              effectiveFaceStatus={effectiveFaceStatus}
              exportFrameProgress={exportFrameProgress}
              topbarReveal={topbarReveal}
              avatarReveal={avatarReveal}
              ratingReveal={ratingReveal}
              potentialReveal={potentialReveal}
              displayedMinimalScore={displayedMinimalScore}
              displayedPotentialScore={displayedPotentialScore}
              ratingBarProgress={ratingBarProgress}
              potentialBarProgress={potentialBarProgress}
              onOpenPicker={onOpenPicker}
              avatarImageRef={avatarImageRef}
              meshCanvasRef={meshCanvasRef}
              onAvatarLoad={() => setAvatarRenderTick((current) => current + 1)}
              showReplay
              onReplay={() => setAnimationNonce((current) => current + 1)}
            />

            <div className="preview-actions" data-export-ignore="true">
              <button
                type="button"
                className="preview-export"
                onClick={handleExportImage}
                disabled={isExporting}
              >
                {isExporting ? 'Exporting...' : 'Export 4K PNG'}
              </button>

              <button
                type="button"
                className="preview-export preview-export--secondary"
                onClick={handleExportVideo}
                disabled={isExporting}
              >
                {isExporting ? 'Exporting...' : 'Export MP4'}
              </button>
            </div>

            {exportMessage && (
              <div className="preview-export-message" data-export-ignore="true">
                {exportMessage}
                {typeof exportProgress === 'number' && (
                  <span className="preview-export-message__progress">{exportProgress}%</span>
                )}
              </div>
            )}

            <div className="export-capture-shell" aria-hidden="true">
              <PreviewCard
                innerRef={exportPreviewCardRef}
                className="preview-card--capture"
                card={card}
                effectiveFaceStatus={effectiveFaceStatus}
                exportFrameProgress={exportFrameProgress}
                topbarReveal={topbarReveal}
                avatarReveal={avatarReveal}
                ratingReveal={ratingReveal}
                potentialReveal={potentialReveal}
                displayedMinimalScore={displayedMinimalScore}
                displayedPotentialScore={displayedPotentialScore}
                ratingBarProgress={ratingBarProgress}
                potentialBarProgress={potentialBarProgress}
                onOpenPicker={onOpenPicker}
                avatarImageRef={exportAvatarImageRef}
                meshCanvasRef={exportMeshCanvasRef}
                onAvatarLoad={() => setAvatarRenderTick((current) => current + 1)}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function DetailedBreakdownEditorScreen({
  card,
  frontFileRef,
  sideFileRef,
  onBack,
  onOpenFrontPicker,
  onOpenSidePicker,
  onFrontImageChange,
  onSideImageChange,
  onChange,
  title = 'Detailed Breakdown',
  description = 'Build the vertical 9:16 breakdown card for creator edits.',
  templateLabel = 'Detailed',
  scoreLabels = ['Overall', 'Harmony', 'Structure', 'Presence', 'Features', 'Symmetry', 'Proportions'],
  scanLabelsDetected,
  detectionMode = 'face',
  enableSideAvatar = false,
  exportBaseName = 'omnimaxx-detailed-breakdown',
}: {
  card: DetailedBreakdownState;
  frontFileRef: React.RefObject<HTMLInputElement | null>;
  sideFileRef?: React.RefObject<HTMLInputElement | null>;
  onBack: () => void;
  onOpenFrontPicker: () => void;
  onOpenSidePicker?: () => void;
  onFrontImageChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onSideImageChange?: (event: ChangeEvent<HTMLInputElement>) => void;
  onChange: <K extends keyof DetailedBreakdownState>(field: K, value: DetailedBreakdownState[K]) => void;
  title?: string;
  description?: string;
  templateLabel?: string;
  scoreLabels?: readonly [string, string, string, string, string, string, string];
  scanLabelsDetected?: readonly [string, string, string, string, string];
  detectionMode?: DetectionMode;
  enableSideAvatar?: boolean;
  exportBaseName?: string;
}) {
  const [animationNonce, setAnimationNonce] = useState(0);
  const [animationDurationMs, setAnimationDurationMs] = useState<AnimationDurationMs>(3000);
  const [animationProgress, setAnimationProgress] = useState(0);
  const [draftFrontOffsetX, setDraftFrontOffsetX] = useState(card.frontImageOffsetX);
  const [draftFrontOffsetY, setDraftFrontOffsetY] = useState(card.frontImageOffsetY);
  const [draftFrontZoom, setDraftFrontZoom] = useState(card.frontImageZoom);
  const [draftSideOffsetX, setDraftSideOffsetX] = useState(card.sideImageOffsetX);
  const [draftSideOffsetY, setDraftSideOffsetY] = useState(card.sideImageOffsetY);
  const [draftSideZoom, setDraftSideZoom] = useState(card.sideImageZoom);
  const [draftFrontPreviewImage, setDraftFrontPreviewImage] = useState<string | null>(card.frontImage);
  const [draftSidePreviewImage, setDraftSidePreviewImage] = useState<string | null>(card.sideImage);
  const [exportFrameProgress, setExportFrameProgress] = useState<number | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const [frontStatus, setFrontStatus] = useState<FaceStatus>('idle');
  const [frontMeshPoints, setFrontMeshPoints] = useState<MeshPoint[]>([]);
  const [sideStatus, setSideStatus] = useState<FaceStatus>('idle');
  const [sideMeshPoints, setSideMeshPoints] = useState<MeshPoint[]>([]);
  const [metrics, setMetrics] = useState<DetailedMetrics | null>(null);
  const [avatarRenderTick, setAvatarRenderTick] = useState(0);
  const heroImageRef = useRef<HTMLImageElement>(null);
  const heroCanvasRef = useRef<HTMLCanvasElement>(null);
  const heroSideImageRef = useRef<HTMLImageElement>(null);
  const heroSideCanvasRef = useRef<HTMLCanvasElement>(null);
  const resultsImageRef = useRef<HTMLImageElement>(null);
  const resultsCanvasRef = useRef<HTMLCanvasElement>(null);
  const resultsSideImageRef = useRef<HTMLImageElement>(null);
  const resultsSideCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportHeroImageRef = useRef<HTMLImageElement>(null);
  const exportHeroCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportHeroSideImageRef = useRef<HTMLImageElement>(null);
  const exportHeroSideCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportResultsImageRef = useRef<HTMLImageElement>(null);
  const exportResultsCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportResultsSideImageRef = useRef<HTMLImageElement>(null);
  const exportResultsSideCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportPreviewCardRef = useRef<HTMLDivElement>(null);
  const effectiveAnimationProgress = exportFrameProgress ?? animationProgress;
  const isFrontCropDirty =
    draftFrontOffsetX !== card.frontImageOffsetX ||
    draftFrontOffsetY !== card.frontImageOffsetY ||
    draftFrontZoom !== card.frontImageZoom;
  const isSideCropDirty =
    draftSideOffsetX !== card.sideImageOffsetX ||
    draftSideOffsetY !== card.sideImageOffsetY ||
    draftSideZoom !== card.sideImageZoom;
  const frontPreviewImage = draftFrontPreviewImage ?? card.frontImage;
  const sidePreviewImage = draftSidePreviewImage ?? card.sideImage;
  const [overallLabel, scoreLabel1, scoreLabel2, scoreLabel3, scoreLabel4, scoreLabel5, scoreLabel6] = scoreLabels;
  const featureRows = [
    [overallLabel, card.overallScore],
    [scoreLabel1, card.harmonyScore],
    [scoreLabel2, card.angularityScore],
    [scoreLabel3, card.dimorphismScore],
    [scoreLabel4, card.featuresScore],
    [scoreLabel5, card.symmetryScore],
    [scoreLabel6, card.proportionsScore],
  ] as const;
  const scanLabels =
    frontStatus === 'detected' && metrics
      ? scanLabelsDetected
        ? [
            `${scanLabelsDetected[0]} ${Math.round(metrics.eyeSpacingPercent)}%`,
            `${scanLabelsDetected[1]} ${Math.round(metrics.jawWidthPercent)}%`,
            `${scanLabelsDetected[2]} ${Math.round(metrics.midfacePercent)}%`,
            `${scanLabelsDetected[3]} ${Math.round(metrics.symmetryPercent)}%`,
            scanLabelsDetected[4],
          ]
        : [
            `Eye spacing ${Math.round(metrics.eyeSpacingPercent)}%`,
            `Jaw width ${Math.round(metrics.jawWidthPercent)}%`,
            `Midface ${Math.round(metrics.midfacePercent)}%`,
            `Symmetry ${Math.round(metrics.symmetryPercent)}%`,
            'Compiling results',
          ]
      : frontStatus === 'no-face'
        ? ['No face detected']
        : frontStatus === 'error'
          ? ['Face scan failed']
          : ['Mapping face', 'Tracking features', 'Scanning ratios'];
  const scanProgress = windowedProgress(effectiveAnimationProgress, 0.1, 0.64);
  const scanIndex = Math.min(scanLabels.length - 1, Math.floor(scanProgress * scanLabels.length));
  const activeScanLabel = scanLabels[scanIndex];
  const analysisReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0, 0.16));
  const analysisFade = 1 - easedProgress(windowedProgress(effectiveAnimationProgress, 0.62, 0.82));
  const analysisBarProgress = easedProgress(windowedProgress(effectiveAnimationProgress, 0.08, 0.64));
  const resultsReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.7, 0.98));
  const displayedOverallScore = card.overallScore * resultsReveal;
  const displayedFeatureRows = featureRows.map(([label, value]) => [label, value * resultsReveal] as const);

  useEffect(() => {
    let frameId = 0;
    const start = performance.now();

    const tick = (now: number) => {
      const next = Math.min((now - start) / animationDurationMs, 1);
      setAnimationProgress(next);

      if (next < 1) {
        frameId = requestAnimationFrame(tick);
      }
    };

    frameId = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(frameId);
  }, [animationNonce, animationDurationMs]);

  useEffect(() => {
    setDraftFrontOffsetX(card.frontImageOffsetX);
    setDraftFrontOffsetY(card.frontImageOffsetY);
    setDraftFrontZoom(card.frontImageZoom);
  }, [card.frontImageOffsetX, card.frontImageOffsetY, card.frontImageZoom, card.frontSourceImage]);

  useEffect(() => {
    setDraftSideOffsetX(card.sideImageOffsetX);
    setDraftSideOffsetY(card.sideImageOffsetY);
    setDraftSideZoom(card.sideImageZoom);
  }, [card.sideImageOffsetX, card.sideImageOffsetY, card.sideImageZoom, card.sideSourceImage]);

  useEffect(() => {
    let cancelled = false;
    if (!card.frontSourceImage) {
      setDraftFrontPreviewImage(card.frontImage);
      return;
    }

    createCroppedAvatarImage({
      source: card.frontSourceImage,
      offsetX: draftFrontOffsetX,
      offsetY: draftFrontOffsetY,
      zoom: draftFrontZoom,
      size: 1400,
    }).then((cropped) => {
      if (!cancelled) {
        setDraftFrontPreviewImage(cropped);
      }
    }).catch(() => {
      if (!cancelled) {
        setDraftFrontPreviewImage(card.frontImage);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [card.frontImage, card.frontSourceImage, draftFrontOffsetX, draftFrontOffsetY, draftFrontZoom]);

  useEffect(() => {
    let cancelled = false;
    if (!card.sideSourceImage) {
      setDraftSidePreviewImage(card.sideImage);
      return;
    }

    createCroppedAvatarImage({
      source: card.sideSourceImage,
      offsetX: draftSideOffsetX,
      offsetY: draftSideOffsetY,
      zoom: draftSideZoom,
      size: 1400,
    }).then((cropped) => {
      if (!cancelled) {
        setDraftSidePreviewImage(cropped);
      }
    }).catch(() => {
      if (!cancelled) {
        setDraftSidePreviewImage(card.sideImage);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [card.sideImage, card.sideSourceImage, draftSideOffsetX, draftSideOffsetY, draftSideZoom]);

  useEffect(() => {
    let cancelled = false;
    let resetFrameId = 0;

    if (!card.frontImage) {
      resetFrameId = requestAnimationFrame(() => {
        if (!cancelled) {
          setFrontStatus('idle');
          setFrontMeshPoints([]);
          setMetrics(null);
        }
      });

      return;
    }

    resetFrameId = requestAnimationFrame(() => {
      if (!cancelled) {
        setFrontStatus('detecting');
        setFrontMeshPoints([]);
        setMetrics(null);
      }
    });

    const image = new Image();
    image.crossOrigin = 'anonymous';

    image.onload = async () => {
      try {
        if (detectionMode === 'body') {
          const detector = await getPoseDetector();
          const poses = await detector.estimatePoses(image, {
            flipHorizontal: false,
            maxPoses: 1,
          });

          if (cancelled) {
            return;
          }

          if (poses.length === 0 || poses[0].keypoints.length === 0) {
            setFrontStatus('no-face');
            return;
          }

          const points = poses[0].keypoints.map((point) => ({ x: point.x, y: point.y }));
          setFrontMeshPoints(points);
          setMetrics(calculateBodyMetrics(points));
          setFrontStatus('detected');
        } else {
          const detector = await getFaceMeshDetector();
          const faces = await detector.estimateFaces(image, {
            flipHorizontal: false,
            staticImageMode: true,
          });

          if (cancelled) {
            return;
          }

          if (faces.length === 0) {
            setFrontStatus('no-face');
            return;
          }

          const points = faces[0].keypoints.map((point) => ({ x: point.x, y: point.y }));
          setFrontMeshPoints(points);
          setMetrics(calculateDetailedMetrics(points));
          setFrontStatus('detected');
        }
      } catch {
        if (!cancelled) {
          setFrontStatus('error');
        }
      }
    };

    image.onerror = () => {
      if (!cancelled) {
        setFrontStatus('error');
      }
    };

    image.src = card.frontImage;

    return () => {
      cancelled = true;
      cancelAnimationFrame(resetFrameId);
    };
  }, [card.frontImage, detectionMode]);

  useEffect(() => {
    let cancelled = false;
    let resetFrameId = 0;

    if (!enableSideAvatar || !card.sideImage) {
      resetFrameId = requestAnimationFrame(() => {
        if (!cancelled) {
          setSideStatus('idle');
          setSideMeshPoints([]);
        }
      });

      return;
    }

    resetFrameId = requestAnimationFrame(() => {
      if (!cancelled) {
        setSideStatus('detecting');
        setSideMeshPoints([]);
      }
    });

    const image = new Image();
    image.crossOrigin = 'anonymous';

    image.onload = async () => {
      try {
        const detector = await getFaceMeshDetector();
        const faces = await detector.estimateFaces(image, {
          flipHorizontal: false,
          staticImageMode: true,
        });

        if (cancelled) {
          return;
        }

        if (faces.length === 0) {
          setSideMeshPoints(createSyntheticSideProfilePoints({
            xMin: image.naturalWidth * 0.18,
            yMin: image.naturalHeight * 0.12,
            width: image.naturalWidth * 0.64,
            height: image.naturalHeight * 0.84,
          }));
          setSideStatus('detected');
          return;
        }

        setSideMeshPoints(faces[0].keypoints.map((point) => ({ x: point.x, y: point.y })));
        setSideStatus('detected');
      } catch {
        if (!cancelled) {
          setSideStatus('error');
        }
      }
    };

    image.onerror = () => {
      if (!cancelled) {
        setSideStatus('error');
      }
    };

    image.src = card.sideImage;

    return () => {
      cancelled = true;
      cancelAnimationFrame(resetFrameId);
    };
  }, [card.sideImage, enableSideAvatar]);

  useEffect(() => {
    const overlayStyle: DetailOverlayStyle = {
      color: card.landmarkColor,
      opacity: card.landmarkOpacity,
      dotSize: card.landmarkDotSize,
      lineThickness: card.landmarkLineThickness,
    };

    const drawOverlay = detectionMode === 'body' ? drawBodyScanOverlay : drawDetailScanOverlay;

    drawOverlay({
      canvas: heroCanvasRef.current,
      image: heroImageRef.current,
      status: frontStatus,
      meshPoints: frontMeshPoints,
      progress: scanProgress,
      overlayStyle,
    });

    drawSideProfileOverlay({
      canvas: heroSideCanvasRef.current,
      image: heroSideImageRef.current,
      status: sideStatus,
      meshPoints: sideMeshPoints,
      progress: scanProgress,
      overlayStyle,
    });

    drawOverlay({
      canvas: resultsCanvasRef.current,
      image: resultsImageRef.current,
      status: frontStatus,
      meshPoints: frontMeshPoints,
      progress: 1,
      overlayStyle,
    });

    drawSideProfileOverlay({
      canvas: resultsSideCanvasRef.current,
      image: resultsSideImageRef.current,
      status: sideStatus,
      meshPoints: sideMeshPoints,
      progress: 1,
      overlayStyle,
    });

    drawOverlay({
      canvas: exportHeroCanvasRef.current,
      image: exportHeroImageRef.current,
      status: frontStatus,
      meshPoints: frontMeshPoints,
      progress: scanProgress,
      overlayStyle,
    });

    drawSideProfileOverlay({
      canvas: exportHeroSideCanvasRef.current,
      image: exportHeroSideImageRef.current,
      status: sideStatus,
      meshPoints: sideMeshPoints,
      progress: scanProgress,
      overlayStyle,
    });

    drawOverlay({
      canvas: exportResultsCanvasRef.current,
      image: exportResultsImageRef.current,
      status: frontStatus,
      meshPoints: frontMeshPoints,
      progress: 1,
      overlayStyle,
    });

    drawSideProfileOverlay({
      canvas: exportResultsSideCanvasRef.current,
      image: exportResultsSideImageRef.current,
      status: sideStatus,
      meshPoints: sideMeshPoints,
      progress: 1,
      overlayStyle,
    });
  }, [
    avatarRenderTick,
    card.landmarkColor,
    card.landmarkDotSize,
    card.landmarkLineThickness,
    card.landmarkOpacity,
    detectionMode,
    frontStatus,
    frontMeshPoints,
    sideStatus,
    sideMeshPoints,
    scanProgress,
  ]);

  const handleExportVideo = async () => {
    if (isExporting) {
      return;
    }

    setIsExporting(true);
    setExportMessage(null);
    setExportProgress(0);

    try {
      setExportMessage('Loading MP4 exporter...');
      setExportProgress(5);
      const ffmpeg = await getFFmpeg();
      const ffmpegLogs: string[] = [];
      const logListener = ({ message }: { message: string }) => {
        ffmpegLogs.push(message);
        if (ffmpegLogs.length > 20) {
          ffmpegLogs.shift();
        }
      };
      const progressListener = ({ progress }: { progress: number }) => {
        const normalized = Math.max(0, Math.min(1, progress));
        setExportProgress(70 + Math.round(normalized * 25));
        setExportMessage(`Encoding MP4... ${Math.round(normalized * 100)}%`);
      };

      ffmpeg.on('log', logListener);
      ffmpeg.on('progress', progressListener);
      setExportProgress(12);

      const sourceNode = exportPreviewCardRef.current;
      if (!sourceNode) {
        throw new Error('Could not prepare the export preview.');
      }

      await waitForExportNodeReady(sourceNode);
      setAnimationNonce((current) => current + 1);
      await wait(80);
      setExportFrameProgress(0);
      await waitForAnimationFrame();
      await waitForAnimationFrame();

      const width = EXPORT_PORTRAIT_LAYOUT_WIDTH;
      const height = EXPORT_PORTRAIT_LAYOUT_HEIGHT;
      const outputWidth = EXPORT_PORTRAIT_VIDEO_WIDTH;
      const outputHeight = EXPORT_PORTRAIT_VIDEO_HEIGHT;
      const fps = 30;
      const totalFrames = Math.max(2, Math.round((animationDurationMs / 1000) * fps));
      const exportId = `${Date.now()}`;
      const inputPattern = `detail-frame-${exportId}-%03d.png`;
      const outputName = `${exportBaseName}-${exportId}.mp4`;

      setExportMessage('Rendering frames...');
      setExportProgress(18);
      const overlayStyle: DetailOverlayStyle = {
        color: card.landmarkColor,
        opacity: card.landmarkOpacity,
        dotSize: card.landmarkDotSize,
        lineThickness: card.landmarkLineThickness,
      };
      const drawOverlay = detectionMode === 'body' ? drawBodyScanOverlay : drawDetailScanOverlay;

      for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        const frameProgress = frameIndex / (totalFrames - 1);
        setExportFrameProgress(frameProgress);
        await waitForAnimationFrame();
        drawOverlay({
          canvas: exportHeroCanvasRef.current,
          image: exportHeroImageRef.current,
          status: frontStatus,
          meshPoints: frontMeshPoints,
          progress: easedProgress(windowedProgress(frameProgress, 0.08, 0.6)),
          overlayStyle,
        });

        drawSideProfileOverlay({
          canvas: exportHeroSideCanvasRef.current,
          image: exportHeroSideImageRef.current,
          status: sideStatus,
          meshPoints: sideMeshPoints,
          progress: easedProgress(windowedProgress(frameProgress, 0.08, 0.6)),
          overlayStyle,
        });

        drawOverlay({
          canvas: exportResultsCanvasRef.current,
          image: exportResultsImageRef.current,
          status: frontStatus,
          meshPoints: frontMeshPoints,
          progress: 1,
          overlayStyle,
        });

        drawSideProfileOverlay({
          canvas: exportResultsSideCanvasRef.current,
          image: exportResultsSideImageRef.current,
          status: sideStatus,
          meshPoints: sideMeshPoints,
          progress: 1,
          overlayStyle,
        });
        await waitForAnimationFrame();

        const frame = await toCanvas(sourceNode, {
          cacheBust: true,
          width,
          height,
          canvasWidth: outputWidth,
          canvasHeight: outputHeight,
          pixelRatio: 1,
          backgroundColor: '#000000',
          skipAutoScale: true,
          style: {
            width: `${width}px`,
            height: `${height}px`,
          },
          filter: (node) => !(node instanceof HTMLElement && node.dataset.exportIgnore === 'true'),
        });
        const blob = await canvasToBlob(frame);
        const fileName = `detail-frame-${exportId}-${frameIndex.toString().padStart(3, '0')}.png`;
        await ffmpeg.writeFile(fileName, new Uint8Array(await blob.arrayBuffer()));
        const renderProgress = (frameIndex + 1) / totalFrames;
        setExportProgress(18 + Math.round(renderProgress * 47));
        setExportMessage(`Rendering frames... ${Math.round(renderProgress * 100)}%`);
      }

      setExportFrameProgress(1);
      await waitForAnimationFrame();

      setExportMessage('Encoding MP4...');
      setExportProgress(70);
      let exitCode = await ffmpeg.exec([
        '-framerate',
        String(fps),
        '-i',
        inputPattern,
        '-c:v',
        'libx264',
        '-preset',
        'slower',
        '-crf',
        '15',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        outputName,
      ]);

      if (exitCode !== 0) {
        exitCode = await ffmpeg.exec([
          '-framerate',
          String(fps),
          '-i',
          inputPattern,
          '-c:v',
          'mpeg4',
          '-q:v',
          '2',
          '-pix_fmt',
          'yuv420p',
          '-movflags',
          '+faststart',
          outputName,
        ]);
      }

      if (exitCode !== 0) {
        const tail = ffmpegLogs.slice(-3).join(' | ').trim();
        throw new Error(tail ? `FFmpeg failed: ${tail}` : `FFmpeg failed with code ${exitCode}.`);
      }

      const data = await ffmpeg.readFile(outputName);
      const bytes = data instanceof Uint8Array ? data : new TextEncoder().encode(String(data));
      const safeBytes = new Uint8Array(bytes.length);
      safeBytes.set(bytes);
      const blob = new Blob([safeBytes], { type: 'video/mp4' });
      downloadBlob(blob, outputName);
      setExportMessage('MP4 exported.');
      setExportProgress(100);

      for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        const fileName = `detail-frame-${exportId}-${frameIndex.toString().padStart(3, '0')}.png`;
        await ffmpeg.deleteFile(fileName);
      }

      await ffmpeg.deleteFile(outputName);
      ffmpeg.off('log', logListener);
      ffmpeg.off('progress', progressListener);
    } catch (error) {
      setExportMessage(describeExportError(error));
      setExportProgress(null);
    } finally {
      setExportFrameProgress(null);
      setIsExporting(false);
    }
  };

  const handleExportImage = async () => {
    if (isExporting) {
      return;
    }

    setIsExporting(true);
    setExportMessage('Rendering 4K PNG...');
    setExportProgress(0);

    try {
      setExportFrameProgress(1);
      await wait(80);
      await waitForAnimationFrame();

      const sourceNode = exportPreviewCardRef.current;
      if (!sourceNode) {
        throw new Error('Could not prepare the export preview.');
      }

      const outputName = `${exportBaseName}-${Date.now()}.png`;
      await waitForExportNodeReady(sourceNode);
      const overlayStyle: DetailOverlayStyle = {
        color: card.landmarkColor,
        opacity: card.landmarkOpacity,
        dotSize: card.landmarkDotSize,
        lineThickness: card.landmarkLineThickness,
      };
      const drawOverlay = detectionMode === 'body' ? drawBodyScanOverlay : drawDetailScanOverlay;

      drawOverlay({
        canvas: exportHeroCanvasRef.current,
        image: exportHeroImageRef.current,
        status: frontStatus,
        meshPoints: frontMeshPoints,
        progress: scanProgress,
        overlayStyle,
      });

      drawSideProfileOverlay({
        canvas: exportHeroSideCanvasRef.current,
        image: exportHeroSideImageRef.current,
        status: sideStatus,
        meshPoints: sideMeshPoints,
        progress: scanProgress,
        overlayStyle,
      });

      drawOverlay({
        canvas: exportResultsCanvasRef.current,
        image: exportResultsImageRef.current,
        status: frontStatus,
        meshPoints: frontMeshPoints,
        progress: 1,
        overlayStyle,
      });

      drawSideProfileOverlay({
        canvas: exportResultsSideCanvasRef.current,
        image: exportResultsSideImageRef.current,
        status: sideStatus,
        meshPoints: sideMeshPoints,
        progress: 1,
        overlayStyle,
      });
      await waitForAnimationFrame();

      const frame = await renderExportCanvas({
        node: sourceNode,
        width: EXPORT_PORTRAIT_LAYOUT_WIDTH,
        height: EXPORT_PORTRAIT_LAYOUT_HEIGHT,
        canvasWidth: EXPORT_PORTRAIT_VIDEO_WIDTH,
        canvasHeight: EXPORT_PORTRAIT_VIDEO_HEIGHT,
      });

      setExportProgress(75);
      const upscaledFrame = upscaleCanvas(frame, EXPORT_PORTRAIT_IMAGE_WIDTH, EXPORT_PORTRAIT_IMAGE_HEIGHT);
      const blob = await canvasToBlob(upscaledFrame);
      downloadBlob(blob, outputName);
      setExportMessage('4K PNG exported.');
      setExportProgress(100);
    } catch (error) {
      setExportMessage(describeExportError(error));
      setExportProgress(null);
    } finally {
      setExportFrameProgress(null);
      setIsExporting(false);
    }
  };

  return (
    <div className="editor-shell">
      <button className="back-button" onClick={onBack}>
        <ArrowLeft size={16} />
        <span>Back</span>
      </button>

      <div className="editor-header">
        <h1>{title}</h1>
        <p>{description}</p>
      </div>

      <div className="editor-layout">
        <aside className="editor-left">
          <div className="template-pill">{templateLabel}</div>

          <div className="compact-grid compact-grid--double detailed-top-grid">
            <Field label="Front Image">
              <button className="control control--upload" onClick={onOpenFrontPicker}>
                <span>{card.frontImage ? 'Change image' : 'Choose file'}</span>
              </button>
              <input
                ref={frontFileRef}
                type="file"
                accept="image/*"
                hidden
                onChange={onFrontImageChange}
              />
              <span className={`face-status face-status--${card.frontImage ? frontStatus : 'idle'}`}>
                {getDetectionStatusLabel(detectionMode, card.frontImage ? frontStatus : 'idle')}
              </span>
            </Field>

            {enableSideAvatar && sideFileRef && onOpenSidePicker && onSideImageChange && (
              <Field label="Side Image">
                <button className="control control--upload" onClick={onOpenSidePicker}>
                  <span>{card.sideImage ? 'Change image' : 'Choose file'}</span>
                </button>
                <input
                  ref={sideFileRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={onSideImageChange}
                />
                <span className={`face-status face-status--${card.sideImage ? sideStatus : 'idle'}`}>
                  {getFaceStatusLabel(card.sideImage ? sideStatus : 'idle')}
                </span>
              </Field>
            )}

            <Field label="Animation Length">
              <select
                className="control"
                value={animationDurationMs}
                onChange={(event) => setAnimationDurationMs(Number(event.target.value) as AnimationDurationMs)}
              >
                <option value={1500}>1.5 sec</option>
                <option value={3000}>3 sec</option>
              </select>
            </Field>

            {enableSideAvatar && (
              <Field label="Layout Swap">
                <label className="check-control">
                  <input
                    type="checkbox"
                    checked={card.swapHeaderWithAvatars}
                    onChange={(event) => onChange('swapHeaderWithAvatars', event.target.checked)}
                  />
                  <span>Put avatars on top</span>
                </label>
              </Field>
            )}
          </div>

          <div className="editor-section">
            <h2>Scores</h2>

            <div className="compact-grid compact-grid--double">
              <FeatureInput
                label={overallLabel}
                value={card.overallScore}
                onChange={(value) => onChange('overallScore', value)}
              />
              <FeatureInput
                label={scoreLabel1}
                value={card.harmonyScore}
                onChange={(value) => onChange('harmonyScore', value)}
              />
              <FeatureInput
                label={scoreLabel2}
                value={card.angularityScore}
                onChange={(value) => onChange('angularityScore', value)}
              />
              <FeatureInput
                label={scoreLabel3}
                value={card.dimorphismScore}
                onChange={(value) => onChange('dimorphismScore', value)}
              />
              <FeatureInput
                label={scoreLabel4}
                value={card.featuresScore}
                onChange={(value) => onChange('featuresScore', value)}
              />
              <FeatureInput
                label={scoreLabel5}
                value={card.symmetryScore}
                onChange={(value) => onChange('symmetryScore', value)}
              />
              <FeatureInput
                label={scoreLabel6}
                value={card.proportionsScore}
                onChange={(value) => onChange('proportionsScore', value)}
              />
            </div>
          </div>

          <div className="editor-section">
            <h2>Colors</h2>

            <div className="compact-grid compact-grid--double">
              <ColorField
                label="Accent Start"
                value={card.accentStart}
                onChange={(value) => onChange('accentStart', value)}
              />
              <ColorField
                label="Accent End"
                value={card.accentEnd}
                onChange={(value) => onChange('accentEnd', value)}
              />
              <ColorField
                label="Number Color"
                value={card.scoreColor}
                onChange={(value) => onChange('scoreColor', value)}
              />
              <ColorField
                label="Landmark Color"
                value={card.landmarkColor}
                onChange={(value) => onChange('landmarkColor', value)}
              />
              <ColorField
                label="Badge Border Start"
                value={card.badgeBorderStart}
                onChange={(value) => onChange('badgeBorderStart', value)}
              />
              <ColorField
                label="Badge Border End"
                value={card.badgeBorderEnd}
                onChange={(value) => onChange('badgeBorderEnd', value)}
              />
            </div>
          </div>

          <div className="editor-section">
            <h2>Landmarks</h2>

            <div className="compact-grid compact-grid--double">
              <TuneInput
                label="Landmark Opacity"
                value={card.landmarkOpacity}
                min={0.1}
                max={1}
                step={0.01}
                onChange={(value) => onChange('landmarkOpacity', value)}
              />
              <TuneInput
                label="Dot Size"
                value={card.landmarkDotSize}
                min={1}
                max={6}
                step={0.1}
                onChange={(value) => onChange('landmarkDotSize', value)}
              />
              <TuneInput
                label="Line Thickness"
                value={card.landmarkLineThickness}
                min={0.5}
                max={4}
                step={0.1}
                onChange={(value) => onChange('landmarkLineThickness', value)}
              />
            </div>
          </div>

          {card.frontSourceImage && (
            <div className="editor-section">
              <div className="avatar-crop-card">
                <div className="avatar-crop-card__header">
                  <h2>Front crop</h2>
                  <button
                    type="button"
                    className="control control--ghost control--ghost-small"
                    onClick={() => {
                      setDraftFrontZoom(1);
                      setDraftFrontOffsetX(0);
                      setDraftFrontOffsetY(0);
                    }}
                  >
                    Reset
                  </button>
                </div>

                <div className="avatar-crop-card__preview">
                  <div
                    className="avatar-crop-card__ring"
                    style={{
                      background: `linear-gradient(135deg, ${card.badgeBorderStart} 0%, ${card.badgeBorderEnd} 100%)`,
                      boxShadow: `0 0 24px ${card.badgeBorderStart}33, 0 0 36px ${card.badgeBorderEnd}1f`,
                    }}
                  >
                    <div className="avatar-crop-card__avatar">
                      <img
                        src={frontPreviewImage ?? card.frontSourceImage}
                        alt="Front crop preview"
                        className="avatar-crop-card__image"
                        draggable={false}
                      />
                    </div>
                  </div>
                </div>

                <div className="avatar-crop-card__controls">
                  <CropSlider label="Zoom" valueLabel={`${draftFrontZoom.toFixed(2)}x`} value={draftFrontZoom} min={1} max={3} step={0.01} onChange={setDraftFrontZoom} />
                  <CropSlider label="Horizontal" valueLabel={`${Math.round(draftFrontOffsetX)}px`} value={draftFrontOffsetX} min={-150} max={150} step={1} onChange={setDraftFrontOffsetX} />
                  <CropSlider label="Vertical" valueLabel={`${Math.round(draftFrontOffsetY)}px`} value={draftFrontOffsetY} min={-150} max={150} step={1} onChange={setDraftFrontOffsetY} />
                  <button
                    type="button"
                    className="preview-export"
                    onClick={async () => {
                      if (!card.frontSourceImage) {
                        return;
                      }

                      const croppedImage = await createCroppedAvatarImage({
                        source: card.frontSourceImage,
                        offsetX: draftFrontOffsetX,
                        offsetY: draftFrontOffsetY,
                        zoom: draftFrontZoom,
                        size: 1400,
                      });

                      onChange('frontImageOffsetX', draftFrontOffsetX);
                      onChange('frontImageOffsetY', draftFrontOffsetY);
                      onChange('frontImageZoom', draftFrontZoom);
                      onChange('frontImage', croppedImage);
                      setDraftFrontPreviewImage(croppedImage);
                      setAnimationNonce((current) => current + 1);
                    }}
                  >
                    Render landmarks
                  </button>
                </div>
              </div>
            </div>
          )}

          {enableSideAvatar && card.sideSourceImage && (
            <div className="editor-section">
              <div className="avatar-crop-card">
                <div className="avatar-crop-card__header">
                  <h2>Side crop</h2>
                  <button
                    type="button"
                    className="control control--ghost control--ghost-small"
                    onClick={() => {
                      setDraftSideZoom(1);
                      setDraftSideOffsetX(0);
                      setDraftSideOffsetY(0);
                    }}
                  >
                    Reset
                  </button>
                </div>

                <div className="avatar-crop-card__preview">
                  <div
                    className="avatar-crop-card__ring"
                    style={{
                      background: `linear-gradient(135deg, ${card.badgeBorderStart} 0%, ${card.badgeBorderEnd} 100%)`,
                      boxShadow: `0 0 24px ${card.badgeBorderStart}33, 0 0 36px ${card.badgeBorderEnd}1f`,
                    }}
                  >
                    <div className="avatar-crop-card__avatar">
                      <img
                        src={sidePreviewImage ?? card.sideSourceImage}
                        alt="Side crop preview"
                        className="avatar-crop-card__image"
                        draggable={false}
                      />
                    </div>
                  </div>
                </div>

                <div className="avatar-crop-card__controls">
                  <CropSlider label="Zoom" valueLabel={`${draftSideZoom.toFixed(2)}x`} value={draftSideZoom} min={1} max={3} step={0.01} onChange={setDraftSideZoom} />
                  <CropSlider label="Horizontal" valueLabel={`${Math.round(draftSideOffsetX)}px`} value={draftSideOffsetX} min={-150} max={150} step={1} onChange={setDraftSideOffsetX} />
                  <CropSlider label="Vertical" valueLabel={`${Math.round(draftSideOffsetY)}px`} value={draftSideOffsetY} min={-150} max={150} step={1} onChange={setDraftSideOffsetY} />
                  <button
                    type="button"
                    className="preview-export"
                    onClick={async () => {
                      if (!card.sideSourceImage) {
                        return;
                      }

                      const croppedImage = await createCroppedAvatarImage({
                        source: card.sideSourceImage,
                        offsetX: draftSideOffsetX,
                        offsetY: draftSideOffsetY,
                        zoom: draftSideZoom,
                        size: 1400,
                      });

                      onChange('sideImageOffsetX', draftSideOffsetX);
                      onChange('sideImageOffsetY', draftSideOffsetY);
                      onChange('sideImageZoom', draftSideZoom);
                      onChange('sideImage', croppedImage);
                      setDraftSidePreviewImage(croppedImage);
                      setAnimationNonce((current) => current + 1);
                    }}
                  >
                    Render landmarks
                  </button>
                </div>
              </div>
            </div>
          )}
        </aside>

        <section className="editor-right">
          <div className="portrait-stage">
            <DetailedPreviewCard
              card={card}
              frontPreviewImage={frontPreviewImage}
              sidePreviewImage={sidePreviewImage}
              suppressFrontOverlay={isFrontCropDirty}
              suppressSideOverlay={isSideCropDirty}
              activeScanLabel={activeScanLabel}
              analysisReveal={analysisReveal}
              analysisFade={analysisFade}
              analysisBarProgress={analysisBarProgress}
              resultsReveal={resultsReveal}
              displayedOverallScore={displayedOverallScore}
              displayedFeatureRows={displayedFeatureRows}
              onOpenFrontPicker={onOpenFrontPicker}
              heroImageRef={heroImageRef}
              heroCanvasRef={heroCanvasRef}
              heroSideImageRef={heroSideImageRef}
              heroSideCanvasRef={heroSideCanvasRef}
              resultsImageRef={resultsImageRef}
              resultsCanvasRef={resultsCanvasRef}
              resultsSideImageRef={resultsSideImageRef}
              resultsSideCanvasRef={resultsSideCanvasRef}
              onImageLoad={() => setAvatarRenderTick((current) => current + 1)}
              avatarShape={templateLabel === 'Height/Posture' ? 'rectangle' : 'circle'}
              enableSideAvatar={enableSideAvatar}
              onOpenSidePicker={onOpenSidePicker}
              showReplay
              onReplay={() => setAnimationNonce((current) => current + 1)}
            />

            <div className="preview-actions preview-actions--portrait" data-export-ignore="true">
              <button
                type="button"
                className="preview-export"
                onClick={handleExportImage}
                disabled={isExporting}
              >
                {isExporting ? 'Exporting...' : 'Export 4K PNG'}
              </button>

              <button
                type="button"
                className="preview-export preview-export--secondary"
                onClick={handleExportVideo}
                disabled={isExporting}
              >
                {isExporting ? 'Exporting...' : 'Export MP4'}
              </button>
            </div>

            {exportMessage && (
              <div className="preview-export-message preview-export-message--portrait" data-export-ignore="true">
                {exportMessage}
                {typeof exportProgress === 'number' && (
                  <span className="preview-export-message__progress">{exportProgress}%</span>
                )}
              </div>
            )}

            <div className="portrait-export-capture-shell" aria-hidden="true">
              <DetailedPreviewCard
                innerRef={exportPreviewCardRef}
                className="portrait-card--capture"
                card={card}
                frontPreviewImage={frontPreviewImage}
                sidePreviewImage={sidePreviewImage}
                activeScanLabel={activeScanLabel}
                analysisReveal={analysisReveal}
                analysisFade={analysisFade}
                analysisBarProgress={analysisBarProgress}
                resultsReveal={resultsReveal}
                displayedOverallScore={displayedOverallScore}
                displayedFeatureRows={displayedFeatureRows}
                onOpenFrontPicker={onOpenFrontPicker}
                heroImageRef={exportHeroImageRef}
                heroCanvasRef={exportHeroCanvasRef}
                heroSideImageRef={exportHeroSideImageRef}
                heroSideCanvasRef={exportHeroSideCanvasRef}
                resultsImageRef={exportResultsImageRef}
                resultsCanvasRef={exportResultsCanvasRef}
                resultsSideImageRef={exportResultsSideImageRef}
                resultsSideCanvasRef={exportResultsSideCanvasRef}
                onImageLoad={() => setAvatarRenderTick((current) => current + 1)}
                avatarShape={templateLabel === 'Height/Posture' ? 'rectangle' : 'circle'}
                enableSideAvatar={enableSideAvatar}
                onOpenSidePicker={onOpenSidePicker}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function DetailedPreviewCard({
  innerRef,
  className,
  card,
  frontPreviewImage,
  sidePreviewImage,
  suppressFrontOverlay,
  suppressSideOverlay,
  activeScanLabel,
  analysisReveal,
  analysisFade,
  analysisBarProgress,
  resultsReveal,
  displayedOverallScore,
  displayedFeatureRows,
  onOpenFrontPicker,
  heroImageRef,
  heroCanvasRef,
  heroSideImageRef,
  heroSideCanvasRef,
  resultsImageRef,
  resultsCanvasRef,
  resultsSideImageRef,
  resultsSideCanvasRef,
  onImageLoad,
  avatarShape = 'circle',
  enableSideAvatar,
  onOpenSidePicker,
  showReplay,
  onReplay,
}: {
  innerRef?: React.RefObject<HTMLDivElement | null>;
  className?: string;
  card: DetailedBreakdownState;
  frontPreviewImage?: string | null;
  sidePreviewImage?: string | null;
  suppressFrontOverlay?: boolean;
  suppressSideOverlay?: boolean;
  activeScanLabel: string;
  analysisReveal: number;
  analysisFade: number;
  analysisBarProgress: number;
  resultsReveal: number;
  displayedOverallScore: number;
  displayedFeatureRows: ReadonlyArray<readonly [string, number]>;
  onOpenFrontPicker: () => void;
  heroImageRef: React.RefObject<HTMLImageElement | null>;
  heroCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  heroSideImageRef?: React.RefObject<HTMLImageElement | null>;
  heroSideCanvasRef?: React.RefObject<HTMLCanvasElement | null>;
  resultsImageRef: React.RefObject<HTMLImageElement | null>;
  resultsCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  resultsSideImageRef?: React.RefObject<HTMLImageElement | null>;
  resultsSideCanvasRef?: React.RefObject<HTMLCanvasElement | null>;
  onImageLoad: () => void;
  avatarShape?: 'circle' | 'rectangle';
  enableSideAvatar?: boolean;
  onOpenSidePicker?: () => void;
  showReplay?: boolean;
  onReplay?: () => void;
}) {
  const swapHeaderWithAvatars = enableSideAvatar && card.swapHeaderWithAvatars;
  const portraitAppbar = (
    <div className={`portrait-appbar ${swapHeaderWithAvatars ? 'portrait-appbar--under-avatar' : ''}`.trim()}>
      <div className="portrait-appbar__brand">
        <img
          src="/omnimaxx-app-icon.png"
          alt="OmniMaxx app icon"
          className="portrait-appbar__logo"
          loading="eager"
          decoding="sync"
          fetchPriority="high"
        />
        <span className="portrait-appbar__name">OMNIMAXX</span>
      </div>

      <div className="portrait-appbar__store">
        <img
          src="/app-store-logo.png"
          alt="App Store"
          className="portrait-appbar__store-image"
          loading="eager"
          decoding="sync"
          fetchPriority="high"
        />
      </div>
    </div>
  );

  return (
    <div
      ref={innerRef}
      className={`portrait-card ${className ?? ''}`.trim()}
      style={{
        ['--detail-accent-start' as string]: card.accentStart,
        ['--detail-accent-end' as string]: card.accentEnd,
        ['--detail-score-color' as string]: card.scoreColor,
        ['--detail-badge-border-start' as string]: card.badgeBorderStart,
        ['--detail-badge-border-end' as string]: card.badgeBorderEnd,
      }}
    >
      {!swapHeaderWithAvatars && (
        <div className="portrait-header">
          {portraitAppbar}
        </div>
      )}

      <div className="portrait-content-stage">
        <div
          className="portrait-analysis-stage"
          style={{
            opacity: analysisFade,
            transform: `translateY(${(1 - analysisReveal) * 20}px) scale(${0.96 + analysisReveal * 0.04})`,
          }}
        >
          <div className={`portrait-shot-stack ${enableSideAvatar ? 'portrait-shot-stack--dual' : ''}`.trim()}>
            <button
              className={`portrait-shot portrait-shot--hero ${avatarShape === 'rectangle' ? 'portrait-shot--hero-rect' : ''} ${enableSideAvatar ? 'portrait-shot--dual' : ''}`.trim()}
              onClick={onOpenFrontPicker}
            >
              {(frontPreviewImage ?? card.frontImage) ? (
                <>
                <img
                  ref={heroImageRef}
                  src={frontPreviewImage ?? card.frontImage ?? undefined}
                  alt="Front profile preview"
                  className="portrait-shot__image"
                  onLoad={onImageLoad}
                  loading="eager"
                  decoding="sync"
                    fetchPriority="high"
                  />
                  {!suppressFrontOverlay && (
                    <canvas
                      ref={heroCanvasRef}
                      className="portrait-shot__overlay"
                      aria-hidden="true"
                    />
                  )}
                </>
              ) : (
                <>
                  <ImagePlus size={26} />
                  <span>{avatarShape === 'rectangle' ? 'Full Body' : 'Front'}</span>
                </>
              )}
              <div className="portrait-shot__badge" aria-hidden="true">
                <img
                  src="/omnimaxx-app-icon.png"
                  alt=""
                  className="portrait-shot__badge-image"
                  loading="eager"
                  decoding="sync"
                  fetchPriority="high"
                />
              </div>
            </button>

            {enableSideAvatar && onOpenSidePicker && (
              <button
                className="portrait-shot portrait-shot--hero portrait-shot--dual portrait-shot--side"
                onClick={onOpenSidePicker}
              >
                {(sidePreviewImage ?? card.sideImage) ? (
                  <>
                    <img
                      ref={heroSideImageRef}
                      src={sidePreviewImage ?? card.sideImage ?? undefined}
                      alt="Side profile preview"
                      className="portrait-shot__image"
                      onLoad={onImageLoad}
                      loading="eager"
                      decoding="sync"
                      fetchPriority="high"
                    />
                    {!suppressSideOverlay && (
                      <canvas
                        ref={heroSideCanvasRef}
                        className="portrait-shot__overlay"
                        aria-hidden="true"
                      />
                    )}
                  </>
                ) : (
                  <>
                    <ImagePlus size={26} />
                    <span>Side</span>
                  </>
                )}
              </button>
            )}

          </div>

          {swapHeaderWithAvatars && portraitAppbar}

          <div className="portrait-analysis-copy">
            <span className="portrait-analysis-copy__eyebrow">Analyzing with OmniMaxx</span>
            <strong>{activeScanLabel}</strong>
          </div>

          <div className="portrait-analysis-bar">
            <div
              className="portrait-analysis-bar__fill"
              style={{
                width: `${analysisBarProgress * 100}%`,
                background: `linear-gradient(90deg, ${card.accentStart} 0%, ${card.accentEnd} 100%)`,
              }}
            />
          </div>
        </div>

        <div
          className="portrait-results-stage"
          style={getRevealStyle(resultsReveal, 24, 0.97)}
        >
          <div className={`portrait-results-avatar-stack ${enableSideAvatar ? 'portrait-results-avatar-stack--dual' : ''}`.trim()}>
            <div className={`portrait-results-avatar ${avatarShape === 'rectangle' ? 'portrait-results-avatar--rect' : ''} ${enableSideAvatar ? 'portrait-results-avatar--dual' : ''}`.trim()}>
              {(frontPreviewImage ?? card.frontImage) ? (
                <>
                  <img
                    ref={resultsImageRef}
                    src={frontPreviewImage ?? card.frontImage ?? undefined}
                    alt="Detailed breakdown avatar"
                    className="portrait-results-avatar__image"
                    onLoad={onImageLoad}
                    loading="eager"
                    decoding="sync"
                    fetchPriority="high"
                  />
                  {!suppressFrontOverlay && (
                    <canvas
                      ref={resultsCanvasRef}
                      className="portrait-results-avatar__overlay"
                      aria-hidden="true"
                    />
                  )}
                </>
              ) : (
                <>
                  <ImagePlus size={24} />
                  <span>Avatar</span>
                </>
              )}
              <div className="portrait-results-avatar__badge" aria-hidden="true">
                <img
                  src="/omnimaxx-app-icon.png"
                  alt=""
                  className="portrait-results-avatar__badge-image"
                  loading="eager"
                  decoding="sync"
                  fetchPriority="high"
                />
              </div>
            </div>

            {enableSideAvatar && (
              <div className="portrait-results-avatar portrait-results-avatar--dual portrait-results-avatar--side">
                {(sidePreviewImage ?? card.sideImage) ? (
                  <>
                    <img
                      ref={resultsSideImageRef}
                      src={sidePreviewImage ?? card.sideImage ?? undefined}
                      alt="Detailed breakdown side avatar"
                      className="portrait-results-avatar__image"
                      onLoad={onImageLoad}
                      loading="eager"
                      decoding="sync"
                      fetchPriority="high"
                    />
                    {!suppressSideOverlay && (
                      <canvas
                        ref={resultsSideCanvasRef}
                        className="portrait-results-avatar__overlay"
                        aria-hidden="true"
                      />
                    )}
                  </>
                ) : (
                  <>
                    <ImagePlus size={24} />
                    <span>Side</span>
                  </>
                )}
              </div>
            )}

          </div>

          {swapHeaderWithAvatars && portraitAppbar}

          <div className="portrait-score-head">
            <span className="portrait-score-head__eyebrow">Overall Score</span>
            <div className="portrait-score-head__value">
              <strong>{displayedOverallScore.toFixed(1)}</strong>
              <span>/10</span>
            </div>
          </div>

          <div className="portrait-panel">
            <div className="portrait-breakdown-grid">
              {displayedFeatureRows.filter(([label]) => label !== 'Overall').map(([label, value]) => (
                <DetailFeatureCell key={label} label={label} value={value} />
              ))}
            </div>
          </div>
        </div>
      </div>

      {showReplay && onReplay && (
        <button
          type="button"
          className="portrait-replay"
          data-export-ignore="true"
          onClick={onReplay}
        >
          Replay
        </button>
      )}
    </div>
  );
}

function AscendEditorScreen({
  card,
  fileRef,
  onBack,
  onOpenPicker,
  onImageChange,
  onChange,
}: {
  card: AscendState;
  fileRef: React.RefObject<HTMLInputElement | null>;
  onBack: () => void;
  onOpenPicker: () => void;
  onImageChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onChange: <K extends keyof AscendState>(field: K, value: AscendState[K]) => void;
}) {
  const [animationNonce, setAnimationNonce] = useState(0);
  const [animationDurationMs, setAnimationDurationMs] = useState<AnimationDurationMs>(1500);
  const [draftImageOffsetX, setDraftImageOffsetX] = useState(card.imageOffsetX);
  const [draftImageOffsetY, setDraftImageOffsetY] = useState(card.imageOffsetY);
  const [draftImageZoom, setDraftImageZoom] = useState(card.imageZoom);
  const [draftPreviewImage, setDraftPreviewImage] = useState<string | null>(card.image);
  const [faceStatus, setFaceStatus] = useState<FaceStatus>('idle');
  const [meshPoints, setMeshPoints] = useState<MeshPoint[]>([]);
  const [avatarRenderTick, setAvatarRenderTick] = useState(0);
  const [animationProgress, setAnimationProgress] = useState(0);
  const [exportFrameProgress, setExportFrameProgress] = useState<number | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const heroImageRef = useRef<HTMLImageElement>(null);
  const meshCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportHeroImageRef = useRef<HTMLImageElement>(null);
  const exportMeshCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportPreviewCardRef = useRef<HTMLDivElement>(null);
  const effectiveAnimationProgress = exportFrameProgress ?? animationProgress;
  const isCropDirty =
    draftImageOffsetX !== card.imageOffsetX ||
    draftImageOffsetY !== card.imageOffsetY ||
    draftImageZoom !== card.imageZoom;
  const previewImage = draftPreviewImage ?? card.image;
  const topbarReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0, 0.16));
  const avatarReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.1, 0.34));
  const primaryReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.24, 0.5));
  const leftReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.42, 0.68));
  const rightReveal = easedProgress(windowedProgress(effectiveAnimationProgress, 0.48, 0.74));
  const pslScoreProgress = easedProgress(windowedProgress(effectiveAnimationProgress, 0.34, 0.88));
  const potentialScoreProgress = easedProgress(windowedProgress(effectiveAnimationProgress, 0.5, 0.92));
  const featuresScoreProgress = easedProgress(windowedProgress(effectiveAnimationProgress, 0.56, 0.96));
  const displayedPslScore = card.pslScore * pslScoreProgress;
  const displayedPotentialScore = card.potentialScore * potentialScoreProgress;
  const displayedFeaturesScore = card.improvementScore * featuresScoreProgress;
  const pslBarProgress =
    (card.pslScore / 8) * 100 * easedProgress(windowedProgress(effectiveAnimationProgress, 0.4, 0.82));
  const potentialBarProgress =
    card.potentialScore * 10 * easedProgress(windowedProgress(effectiveAnimationProgress, 0.54, 0.9));
  const featuresBarProgress =
    card.improvementScore * 10 * easedProgress(windowedProgress(effectiveAnimationProgress, 0.6, 0.94));
  const meshDrawProgress = easedProgress(windowedProgress(effectiveAnimationProgress, 0.16, 0.64));

  useEffect(() => {
    let frameId = 0;
    const start = performance.now();

    const tick = (now: number) => {
      const next = Math.min((now - start) / animationDurationMs, 1);
      setAnimationProgress(next);

      if (next < 1) {
        frameId = requestAnimationFrame(tick);
      }
    };

    frameId = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(frameId);
  }, [animationDurationMs, animationNonce]);

  useEffect(() => {
    setDraftImageOffsetX(card.imageOffsetX);
    setDraftImageOffsetY(card.imageOffsetY);
    setDraftImageZoom(card.imageZoom);
  }, [card.imageOffsetX, card.imageOffsetY, card.imageZoom, card.sourceImage]);

  useEffect(() => {
    let cancelled = false;
    if (!card.sourceImage) {
      setDraftPreviewImage(card.image);
      return;
    }

    createCroppedAvatarImage({
      source: card.sourceImage,
      offsetX: draftImageOffsetX,
      offsetY: draftImageOffsetY,
      zoom: draftImageZoom,
      size: 1400,
    }).then((cropped) => {
      if (!cancelled) {
        setDraftPreviewImage(cropped);
      }
    }).catch(() => {
      if (!cancelled) {
        setDraftPreviewImage(card.image);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [card.image, card.sourceImage, draftImageOffsetX, draftImageOffsetY, draftImageZoom]);

  useEffect(() => {
    let cancelled = false;
    let resetFrameId = 0;

    if (!card.image) {
      resetFrameId = requestAnimationFrame(() => {
        if (!cancelled) {
          setFaceStatus('idle');
          setMeshPoints([]);
        }
      });

      return;
    }

    resetFrameId = requestAnimationFrame(() => {
      if (!cancelled) {
        setFaceStatus('detecting');
        setMeshPoints([]);
      }
    });

    const image = new Image();
    image.crossOrigin = 'anonymous';

    image.onload = async () => {
      try {
        const detector = await getFaceMeshDetector();
        const faces = await detector.estimateFaces(image, {
          flipHorizontal: false,
          staticImageMode: true,
        });

        if (cancelled) {
          return;
        }

        if (faces.length === 0) {
          setFaceStatus('no-face');
          setMeshPoints([]);
          return;
        }

        setFaceStatus('detected');
        setMeshPoints(faces[0].keypoints.map((point) => ({ x: point.x, y: point.y })));
      } catch {
        if (!cancelled) {
          setFaceStatus('error');
          setMeshPoints([]);
        }
      }
    };

    image.onerror = () => {
      if (!cancelled) {
        setFaceStatus('error');
        setMeshPoints([]);
      }
    };

    image.src = card.image;

    return () => {
      cancelled = true;
      cancelAnimationFrame(resetFrameId);
    };
  }, [card.image]);

  useEffect(() => {
    drawFaceMesh({
      canvas: meshCanvasRef.current,
      image: heroImageRef.current,
      status: card.image ? faceStatus : 'idle',
      meshPoints,
      meshDrawProgress,
      meshColor: card.meshColor,
      meshOpacity: card.meshOpacity,
    });

    drawFaceMesh({
      canvas: exportMeshCanvasRef.current,
      image: exportHeroImageRef.current,
      status: card.image ? faceStatus : 'idle',
      meshPoints,
      meshDrawProgress,
      meshColor: card.meshColor,
      meshOpacity: card.meshOpacity,
    });
  }, [
    avatarRenderTick,
    card.image,
    card.meshColor,
    card.meshOpacity,
    faceStatus,
    meshDrawProgress,
    meshPoints,
  ]);

  const handleRenderMesh = async () => {
    if (!card.sourceImage) {
      return;
    }

    const croppedImage = await createCroppedAvatarImage({
      source: card.sourceImage,
      offsetX: draftImageOffsetX,
      offsetY: draftImageOffsetY,
      zoom: draftImageZoom,
      size: 1400,
    });

    onChange('imageOffsetX', draftImageOffsetX);
    onChange('imageOffsetY', draftImageOffsetY);
    onChange('imageZoom', draftImageZoom);
    onChange('image', croppedImage);
    setDraftPreviewImage(croppedImage);
    setAnimationNonce((current) => current + 1);
  };

  const handleExportVideo = async () => {
    if (isExporting) {
      return;
    }

    setIsExporting(true);
    setExportMessage(null);
    setExportProgress(0);

    try {
      setExportMessage('Loading MP4 exporter...');
      setExportProgress(5);
      const ffmpeg = await getFFmpeg();
      const ffmpegLogs: string[] = [];
      const logListener = ({ message }: { message: string }) => {
        ffmpegLogs.push(message);
        if (ffmpegLogs.length > 20) {
          ffmpegLogs.shift();
        }
      };
      const progressListener = ({ progress }: { progress: number }) => {
        const normalized = Math.max(0, Math.min(1, progress));
        setExportProgress(70 + Math.round(normalized * 25));
        setExportMessage(`Encoding MP4... ${Math.round(normalized * 100)}%`);
      };

      ffmpeg.on('log', logListener);
      ffmpeg.on('progress', progressListener);
      setExportProgress(12);

      const sourceNode = exportPreviewCardRef.current;
      if (!sourceNode) {
        throw new Error('Could not prepare the export preview.');
      }

      await waitForExportNodeReady(sourceNode);
      setAnimationNonce((current) => current + 1);
      await wait(80);
      setExportFrameProgress(0);
      await waitForAnimationFrame();
      await waitForAnimationFrame();

      const width = EXPORT_PORTRAIT_LAYOUT_WIDTH;
      const height = EXPORT_PORTRAIT_LAYOUT_HEIGHT;
      const outputWidth = EXPORT_PORTRAIT_VIDEO_WIDTH;
      const outputHeight = EXPORT_PORTRAIT_VIDEO_HEIGHT;
      const fps = 30;
      const totalFrames = Math.max(2, Math.round((animationDurationMs / 1000) * fps));
      const exportId = `${Date.now()}`;
      const inputPattern = `ascend-frame-${exportId}-%03d.png`;
      const outputName = `omnimaxx-ascend-${exportId}.mp4`;

      setExportMessage('Rendering frames...');
      setExportProgress(18);

      for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        const frameProgress = frameIndex / (totalFrames - 1);
        setExportFrameProgress(frameProgress);
        await waitForAnimationFrame();
        drawFaceMesh({
          canvas: exportMeshCanvasRef.current,
          image: exportHeroImageRef.current,
          status: card.image ? faceStatus : 'idle',
          meshPoints,
          meshDrawProgress: easedProgress(windowedProgress(frameProgress, 0.18, 0.74)),
          meshColor: card.meshColor,
          meshOpacity: card.meshOpacity,
        });
        await waitForAnimationFrame();

        const frame = await toCanvas(sourceNode, {
          cacheBust: true,
          width,
          height,
          canvasWidth: outputWidth,
          canvasHeight: outputHeight,
          pixelRatio: 1,
          backgroundColor: '#000000',
          skipAutoScale: true,
          style: {
            width: `${width}px`,
            height: `${height}px`,
          },
          filter: (node) => !(node instanceof HTMLElement && node.dataset.exportIgnore === 'true'),
        });
        const blob = await canvasToBlob(frame);
        const fileName = `ascend-frame-${exportId}-${frameIndex.toString().padStart(3, '0')}.png`;
        await ffmpeg.writeFile(fileName, new Uint8Array(await blob.arrayBuffer()));
        const renderProgress = (frameIndex + 1) / totalFrames;
        setExportProgress(18 + Math.round(renderProgress * 47));
        setExportMessage(`Rendering frames... ${Math.round(renderProgress * 100)}%`);
      }

      setExportFrameProgress(1);
      await waitForAnimationFrame();

      setExportMessage('Encoding MP4...');
      setExportProgress(70);
      let exitCode = await ffmpeg.exec([
        '-framerate',
        String(fps),
        '-i',
        inputPattern,
        '-c:v',
        'libx264',
        '-preset',
        'slower',
        '-crf',
        '15',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        outputName,
      ]);

      if (exitCode !== 0) {
        exitCode = await ffmpeg.exec([
          '-framerate',
          String(fps),
          '-i',
          inputPattern,
          '-c:v',
          'mpeg4',
          '-q:v',
          '2',
          '-pix_fmt',
          'yuv420p',
          '-movflags',
          '+faststart',
          outputName,
        ]);
      }

      if (exitCode !== 0) {
        const tail = ffmpegLogs.slice(-3).join(' | ').trim();
        throw new Error(tail ? `FFmpeg failed: ${tail}` : `FFmpeg failed with code ${exitCode}.`);
      }

      const data = await ffmpeg.readFile(outputName);
      const bytes = data instanceof Uint8Array ? data : new TextEncoder().encode(String(data));
      const safeBytes = new Uint8Array(bytes.length);
      safeBytes.set(bytes);
      const blob = new Blob([safeBytes], { type: 'video/mp4' });
      downloadBlob(blob, outputName);
      setExportMessage('MP4 exported.');
      setExportProgress(100);

      for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        const fileName = `ascend-frame-${exportId}-${frameIndex.toString().padStart(3, '0')}.png`;
        await ffmpeg.deleteFile(fileName);
      }

      await ffmpeg.deleteFile(outputName);
      ffmpeg.off('log', logListener);
      ffmpeg.off('progress', progressListener);
    } catch (error) {
      setExportMessage(describeExportError(error));
      setExportProgress(null);
    } finally {
      setExportFrameProgress(null);
      setIsExporting(false);
    }
  };

  const handleExportImage = async () => {
    if (isExporting) {
      return;
    }

    setIsExporting(true);
    setExportMessage('Rendering 4K PNG...');
    setExportProgress(0);

    try {
      setExportFrameProgress(1);
      await wait(80);
      await waitForAnimationFrame();

      const sourceNode = exportPreviewCardRef.current;
      if (!sourceNode) {
        throw new Error('Could not prepare the export preview.');
      }

      const outputName = `omnimaxx-ascend-${Date.now()}.png`;
      await waitForExportNodeReady(sourceNode);
      drawFaceMesh({
        canvas: exportMeshCanvasRef.current,
        image: exportHeroImageRef.current,
        status: card.image ? faceStatus : 'idle',
        meshPoints,
        meshDrawProgress: 1,
        meshColor: card.meshColor,
        meshOpacity: card.meshOpacity,
      });
      await waitForAnimationFrame();

      const frame = await renderExportCanvas({
        node: sourceNode,
        width: EXPORT_PORTRAIT_LAYOUT_WIDTH,
        height: EXPORT_PORTRAIT_LAYOUT_HEIGHT,
        canvasWidth: EXPORT_PORTRAIT_VIDEO_WIDTH,
        canvasHeight: EXPORT_PORTRAIT_VIDEO_HEIGHT,
      });

      setExportProgress(75);
      const upscaledFrame = upscaleCanvas(frame, EXPORT_PORTRAIT_IMAGE_WIDTH, EXPORT_PORTRAIT_IMAGE_HEIGHT);
      const blob = await canvasToBlob(upscaledFrame);
      downloadBlob(blob, outputName);
      setExportMessage('4K PNG exported.');
      setExportProgress(100);
    } catch (error) {
      setExportMessage(describeExportError(error));
      setExportProgress(null);
    } finally {
      setExportFrameProgress(null);
      setIsExporting(false);
    }
  };

  return (
    <div className="editor-shell">
      <button className="back-button" onClick={onBack}>
        <ArrowLeft size={16} />
        <span>Back</span>
      </button>

      <div className="editor-header">
        <h1>ASCEND</h1>
        <p>Build the 9:16 glow-up projection template.</p>
      </div>

      <div className="editor-layout">
        <aside className="editor-left">
          <div className="template-pill">Ascend</div>

          <div className="compact-grid compact-grid--double detailed-top-grid">
            <Field label="Projection Image">
              <button className="control control--upload" onClick={onOpenPicker}>
                <span>{card.image ? 'Change image' : 'Choose file'}</span>
              </button>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                hidden
                onChange={onImageChange}
              />
              <span className={`face-status face-status--${card.image ? faceStatus : 'idle'}`}>
                {getFaceStatusLabel(card.image ? faceStatus : 'idle')}
              </span>
            </Field>

            <Field label="PSL Rating">
              <select
                className="control"
                value={card.pslTier}
                onChange={(event) => onChange('pslTier', event.target.value)}
              >
                {TIER_OPTIONS.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </Field>

            <Field label="Animation Length">
              <select
                className="control"
                value={animationDurationMs}
                onChange={(event) => setAnimationDurationMs(Number(event.target.value) as AnimationDurationMs)}
              >
                <option value={1500}>1.5 sec</option>
                <option value={3000}>3 sec</option>
              </select>
            </Field>
          </div>

          <div className="editor-section">
            <h2>Scores</h2>

            <div className="compact-grid compact-grid--double">
              <FeatureInput
                label="PSL Score"
                value={card.pslScore}
                max={8}
                onChange={(value) => onChange('pslScore', value)}
              />
              <FeatureInput
                label="Potential Score"
                value={card.potentialScore}
                onChange={(value) => onChange('potentialScore', value)}
              />
              <Field label="Potential Rating">
                <select
                  className="control"
                  value={card.potentialTier}
                  onChange={(event) => onChange('potentialTier', event.target.value)}
                >
                  {TIER_OPTIONS.map((option) => (
                    <option key={option}>{option}</option>
                  ))}
                </select>
              </Field>
              <FeatureInput
                label="Improvement"
                value={card.improvementScore}
                onChange={(value) => onChange('improvementScore', value)}
              />
            </div>
          </div>

          <div className="editor-section">
            <h2>Colors</h2>

            <div className="compact-grid compact-grid--double">
              <ColorField
                label="Accent Start"
                value={card.accentStart}
                onChange={(value) => onChange('accentStart', value)}
              />
              <ColorField
                label="Accent End"
                value={card.accentEnd}
                onChange={(value) => onChange('accentEnd', value)}
              />
              <ColorField
                label="Font Color"
                value={card.fontColor}
                onChange={(value) => onChange('fontColor', value)}
              />
              <ColorField
                label="Avatar Border Start"
                value={card.avatarBorderStart}
                onChange={(value) => onChange('avatarBorderStart', value)}
              />
              <ColorField
                label="Avatar Border End"
                value={card.avatarBorderEnd}
                onChange={(value) => onChange('avatarBorderEnd', value)}
              />
              <ColorField
                label="Mesh Color"
                value={card.meshColor}
                onChange={(value) => onChange('meshColor', value)}
              />
            </div>
          </div>

          <div className="editor-section">
            <h2>Mesh</h2>

            <div className="compact-grid compact-grid--double">
              <TuneInput
                label="Mesh Opacity"
                value={card.meshOpacity}
                min={0.1}
                max={1}
                step={0.01}
                onChange={(value) => onChange('meshOpacity', value)}
              />
            </div>
          </div>

          {card.sourceImage && (
            <div className="editor-section">
              <div className="avatar-crop-card">
                <div className="avatar-crop-card__header">
                  <h2>Avatar crop</h2>
                  <button
                    type="button"
                    className="control control--ghost control--ghost-small"
                    onClick={() => {
                      setDraftImageZoom(1);
                      setDraftImageOffsetX(0);
                      setDraftImageOffsetY(0);
                    }}
                  >
                    Reset
                  </button>
                </div>

                <div className="avatar-crop-card__preview">
                  <div
                    className="avatar-crop-card__ring"
                    style={{
                      background: `linear-gradient(135deg, ${card.avatarBorderStart} 0%, ${card.avatarBorderEnd} 100%)`,
                      boxShadow: `0 0 24px ${card.avatarBorderStart}33, 0 0 36px ${card.avatarBorderEnd}1f`,
                    }}
                  >
                    <div className="avatar-crop-card__avatar">
                      <img
                        src={previewImage ?? card.sourceImage}
                        alt="Ascend crop preview"
                        className="avatar-crop-card__image"
                        draggable={false}
                      />
                    </div>
                  </div>
                </div>

                <div className="avatar-crop-card__controls">
                  <CropSlider label="Zoom" valueLabel={`${draftImageZoom.toFixed(2)}x`} value={draftImageZoom} min={1} max={3} step={0.01} onChange={setDraftImageZoom} />
                  <CropSlider label="Horizontal" valueLabel={`${Math.round(draftImageOffsetX)}px`} value={draftImageOffsetX} min={-150} max={150} step={1} onChange={setDraftImageOffsetX} />
                  <CropSlider label="Vertical" valueLabel={`${Math.round(draftImageOffsetY)}px`} value={draftImageOffsetY} min={-150} max={150} step={1} onChange={setDraftImageOffsetY} />
                  <button type="button" className="preview-export" onClick={handleRenderMesh}>
                    Render mesh
                  </button>
                </div>
              </div>
            </div>
          )}
        </aside>

        <section className="editor-right">
          <div className="portrait-stage">
            <AscendPreviewCard
              card={card}
              previewImage={previewImage}
              suppressMesh={isCropDirty}
              topbarReveal={topbarReveal}
              avatarReveal={avatarReveal}
              primaryReveal={primaryReveal}
              leftReveal={leftReveal}
              rightReveal={rightReveal}
              displayedPslScore={displayedPslScore}
              displayedPotentialScore={displayedPotentialScore}
              displayedFeaturesScore={displayedFeaturesScore}
              pslBarProgress={pslBarProgress}
              potentialBarProgress={potentialBarProgress}
              featuresBarProgress={featuresBarProgress}
              onOpenPicker={onOpenPicker}
              heroImageRef={heroImageRef}
              meshCanvasRef={meshCanvasRef}
              onHeroLoad={() => setAvatarRenderTick((current) => current + 1)}
              effectiveFaceStatus={card.image ? faceStatus : 'idle'}
              showReplay
              onReplay={() => setAnimationNonce((current) => current + 1)}
            />

            <div className="preview-actions preview-actions--portrait" data-export-ignore="true">
              <button
                type="button"
                className="preview-export"
                onClick={handleExportImage}
                disabled={isExporting}
              >
                {isExporting ? 'Exporting...' : 'Export 4K PNG'}
              </button>

              <button
                type="button"
                className="preview-export preview-export--secondary"
                onClick={handleExportVideo}
                disabled={isExporting}
              >
                {isExporting ? 'Exporting...' : 'Export MP4'}
              </button>
            </div>

            {exportMessage && (
              <div className="preview-export-message preview-export-message--portrait" data-export-ignore="true">
                {exportMessage}
                {typeof exportProgress === 'number' && (
                  <span className="preview-export-message__progress">{exportProgress}%</span>
                )}
              </div>
            )}

            <div className="portrait-export-capture-shell" aria-hidden="true">
              <AscendPreviewCard
                innerRef={exportPreviewCardRef}
                className="ascend-card--capture"
                card={card}
                previewImage={previewImage}
                topbarReveal={topbarReveal}
                avatarReveal={avatarReveal}
                primaryReveal={primaryReveal}
                leftReveal={leftReveal}
                rightReveal={rightReveal}
                displayedPslScore={displayedPslScore}
                displayedPotentialScore={displayedPotentialScore}
                displayedFeaturesScore={displayedFeaturesScore}
                pslBarProgress={pslBarProgress}
                potentialBarProgress={potentialBarProgress}
                featuresBarProgress={featuresBarProgress}
                onOpenPicker={onOpenPicker}
                heroImageRef={exportHeroImageRef}
                meshCanvasRef={exportMeshCanvasRef}
                onHeroLoad={() => setAvatarRenderTick((current) => current + 1)}
                effectiveFaceStatus={card.image ? faceStatus : 'idle'}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function AscendPreviewCard({
  innerRef,
  className,
  card,
  previewImage,
  suppressMesh,
  topbarReveal,
  avatarReveal,
  primaryReveal,
  leftReveal,
  rightReveal,
  displayedPslScore,
  displayedPotentialScore,
  displayedFeaturesScore,
  pslBarProgress,
  potentialBarProgress,
  featuresBarProgress,
  onOpenPicker,
  heroImageRef,
  meshCanvasRef,
  onHeroLoad,
  effectiveFaceStatus,
  showReplay,
  onReplay,
}: {
  innerRef?: React.RefObject<HTMLDivElement | null>;
  className?: string;
  card: AscendState;
  previewImage?: string | null;
  suppressMesh?: boolean;
  topbarReveal: number;
  avatarReveal: number;
  primaryReveal: number;
  leftReveal: number;
  rightReveal: number;
  displayedPslScore: number;
  displayedPotentialScore: number;
  displayedFeaturesScore: number;
  pslBarProgress: number;
  potentialBarProgress: number;
  featuresBarProgress: number;
  onOpenPicker: () => void;
  heroImageRef: React.RefObject<HTMLImageElement | null>;
  meshCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  onHeroLoad: () => void;
  effectiveFaceStatus: FaceStatus;
  showReplay?: boolean;
  onReplay?: () => void;
}) {
  return (
    <div
      ref={innerRef}
      className={`ascend-card ${className ?? ''}`.trim()}
      style={{
        ['--ascend-accent-start' as string]: card.accentStart,
        ['--ascend-accent-end' as string]: card.accentEnd,
        ['--ascend-font-color' as string]: card.fontColor,
        ['--ascend-avatar-border-start' as string]: card.avatarBorderStart,
        ['--ascend-avatar-border-end' as string]: card.avatarBorderEnd,
      }}
    >
      <div
        className="portrait-header"
        style={getRevealStyle(topbarReveal, 12, 0.98)}
      >
        <div className="portrait-appbar">
          <div className="portrait-appbar__brand">
            <img
              src="/omnimaxx-app-icon.png"
              alt="OmniMaxx app icon"
              className="portrait-appbar__logo"
              loading="eager"
              decoding="sync"
              fetchPriority="high"
            />
            <span className="portrait-appbar__name">OMNIMAXX</span>
          </div>

          <div className="portrait-appbar__store">
            <img
              src="/app-store-logo.png"
              alt="App Store"
              className="portrait-appbar__store-image"
              loading="eager"
              decoding="sync"
              fetchPriority="high"
            />
          </div>
        </div>
      </div>

      <div className="ascend-hero-wrap" style={getRevealStyle(avatarReveal, 12, 0.94)}>
        <button className="ascend-hero" onClick={onOpenPicker}>
          {(previewImage ?? card.image) ? (
            <>
              <img
                ref={heroImageRef}
                src={previewImage ?? card.image ?? undefined}
                alt="Ascend preview"
                className="ascend-hero__image"
                onLoad={onHeroLoad}
                loading="eager"
                decoding="sync"
                fetchPriority="high"
              />
              {effectiveFaceStatus === 'detected' && !suppressMesh && (
                <canvas
                  ref={meshCanvasRef}
                  className="ascend-hero__mesh"
                  aria-hidden="true"
                />
              )}
            </>
          ) : (
            <>
              <ImagePlus size={34} />
              <span>Projection Image</span>
            </>
          )}
          <div className="ascend-hero__badge" aria-hidden="true">
            <img
              src="/omnimaxx-app-icon.png"
              alt=""
              className="ascend-hero__badge-image"
              loading="eager"
              decoding="sync"
              fetchPriority="high"
            />
          </div>
        </button>
      </div>

      <div className="ascend-score-stack">
        <div className="ascend-score-card ascend-score-card--primary" style={getRevealStyle(primaryReveal, 24, 0.96)}>
          <div className="ascend-score-card__header">
            <div>
              <span className="ascend-score-card__label">PSL Score</span>
              <small>Facial rating</small>
            </div>
            <strong className="ascend-score-card__tier">{card.pslTier}</strong>
          </div>

          <div className="ascend-score-card__main">
            <span className="ascend-score-card__value">{displayedPslScore.toFixed(1)}</span>
            <span className="ascend-score-card__unit">/8</span>
          </div>

          <div className="ascend-score-card__track">
            <div
              className="ascend-score-card__fill"
              style={{ width: `${pslBarProgress}%` }}
            />
          </div>
        </div>

        <div className="ascend-score-grid">
          <div className="ascend-score-card ascend-score-card--secondary" style={getRevealStyle(leftReveal, 24, 0.96)}>
            <span className="ascend-score-card__label">Potential</span>
            <strong className="ascend-score-card__tier ascend-score-card__tier--small">
              {card.potentialTier}
            </strong>
            <span className="ascend-score-card__value ascend-score-card__value--compact">
              {displayedPotentialScore.toFixed(1)}
            </span>
            <div className="ascend-score-card__track ascend-score-card__track--compact">
              <div
                className="ascend-score-card__fill"
                style={{ width: `${potentialBarProgress}%` }}
              />
            </div>
          </div>

          <div className="ascend-score-card ascend-score-card--secondary" style={getRevealStyle(rightReveal, 24, 0.96)}>
            <span className="ascend-score-card__label">Features</span>
            <strong className="ascend-score-card__tier ascend-score-card__tier--small">
              Score
            </strong>
            <span className="ascend-score-card__value ascend-score-card__value--compact">
              {displayedFeaturesScore.toFixed(1)}
            </span>
            <div className="ascend-score-card__track ascend-score-card__track--compact">
              <div
                className="ascend-score-card__fill"
                style={{ width: `${featuresBarProgress}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {showReplay && onReplay && (
        <button
          type="button"
          className="portrait-replay"
          data-export-ignore="true"
          onClick={onReplay}
        >
          Replay
        </button>
      )}
    </div>
  );
}

function PreviewCard({
  innerRef,
  className,
  card,
  previewImage,
  suppressMesh,
  effectiveFaceStatus,
  exportFrameProgress,
  topbarReveal,
  avatarReveal,
  ratingReveal,
  potentialReveal,
  displayedMinimalScore,
  displayedPotentialScore,
  ratingBarProgress,
  potentialBarProgress,
  onOpenPicker,
  avatarImageRef,
  meshCanvasRef,
  onAvatarLoad,
  showReplay,
  onReplay,
}: {
  innerRef?: React.RefObject<HTMLDivElement | null>;
  className?: string;
  card: ScorecardState;
  previewImage?: string | null;
  suppressMesh?: boolean;
  effectiveFaceStatus: FaceStatus;
  exportFrameProgress: number | null;
  topbarReveal: number;
  avatarReveal: number;
  ratingReveal: number;
  potentialReveal: number;
  displayedMinimalScore: number;
  displayedPotentialScore: number;
  ratingBarProgress: number;
  potentialBarProgress: number;
  onOpenPicker: () => void;
  avatarImageRef: React.RefObject<HTMLImageElement | null>;
  meshCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  onAvatarLoad: () => void;
  showReplay?: boolean;
  onReplay?: () => void;
}) {
  return (
    <div ref={innerRef} className={`preview-card ${className ?? ''}`.trim()}>
      <div
        className={`preview-topbar ${exportFrameProgress === null ? 'preview-topbar--animated' : ''}`}
        style={exportFrameProgress === null ? undefined : getRevealStyle(topbarReveal, 12, 0.98)}
      >
        <div className="preview-brand">
          <img
            src="/omnimaxx-app-icon.png"
            alt="OmniMaxx app icon"
            className="preview-brand__logo"
            loading="eager"
            decoding="sync"
            fetchPriority="high"
          />
          <span className="preview-brand__name">OMNIMAXX</span>
        </div>

        <div className="preview-store">
          <img
            src="/app-store-logo.png"
            alt="App Store"
            className="preview-store__image"
            loading="eager"
            decoding="sync"
            fetchPriority="high"
          />
        </div>
      </div>

      <div className="preview-avatar-wrap">
        <div
          className={`preview-avatar-ring ${exportFrameProgress === null ? 'preview-avatar-ring--animated' : ''}`}
          style={{
            background: `linear-gradient(135deg, ${card.avatarBorderStart} 0%, ${card.avatarBorderEnd} 100%)`,
            boxShadow: `0 0 34px ${card.avatarBorderStart}33, 0 0 52px ${card.avatarBorderEnd}1f`,
            ['--avatar-glow-start' as string]: `${card.avatarBorderStart}55`,
            ['--avatar-glow-end' as string]: `${card.avatarBorderEnd}10`,
            ['--badge-border-start' as string]: card.avatarBorderStart,
            ['--badge-border-end' as string]: card.avatarBorderEnd,
            ...(exportFrameProgress === null ? {} : getRevealStyle(avatarReveal, 10, 0.92)),
          }}
        >
          <button className="preview-avatar" onClick={onOpenPicker}>
            {(previewImage ?? card.image) ? (
              <>
                <div className="preview-avatar__content">
                <img
                  ref={avatarImageRef}
                  src={previewImage ?? card.image ?? undefined}
                  alt="Avatar preview"
                  className="preview-avatar__image"
                  onLoad={onAvatarLoad}
                  loading="eager"
                  decoding="sync"
                    fetchPriority="high"
                  />
                  {effectiveFaceStatus === 'detected' && !suppressMesh && (
                    <canvas
                      ref={meshCanvasRef}
                      className="preview-avatar__mesh"
                      aria-hidden="true"
                    />
                  )}
                </div>
              </>
            ) : (
              <>
                <ImagePlus size={28} />
                <span>Insert Image</span>
              </>
            )}
            <div className="preview-avatar__badge" aria-hidden="true">
              <img
                src="/omnimaxx-app-icon.png"
                alt=""
                className="preview-avatar__badge-image"
                loading="eager"
                decoding="sync"
                fetchPriority="high"
              />
            </div>
          </button>
        </div>
      </div>

      <div className={`preview-grid ${exportFrameProgress === null ? 'preview-grid--animated' : ''}`}>
        <ScoreBox
          label="RATING"
          score={displayedMinimalScore}
          tier={card.minimalTier}
          gradientStart={card.ratingGradientStart}
          gradientEnd={card.ratingGradientEnd}
          barStart={card.ratingBarStart}
          barEnd={card.ratingBarEnd}
          barWidth={ratingBarProgress}
          revealStyle={exportFrameProgress === null ? undefined : getRevealStyle(ratingReveal, 26, 0.96)}
        />
        <ScoreBox
          label="POTENTIAL"
          score={displayedPotentialScore}
          tier={card.potentialTier}
          gradientStart={card.potentialGradientStart}
          gradientEnd={card.potentialGradientEnd}
          barStart={card.potentialBarStart}
          barEnd={card.potentialBarEnd}
          barWidth={potentialBarProgress}
          revealStyle={exportFrameProgress === null ? undefined : getRevealStyle(potentialReveal, 26, 0.96)}
        />
      </div>

      {showReplay && onReplay && (
        <button
          type="button"
          className="preview-replay"
          data-export-ignore="true"
          onClick={onReplay}
        >
          Replay
        </button>
      )}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      {children}
    </label>
  );
}

function ColorField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="field field--color">
      <span className="field__label">{label}</span>
      <div className="color-control">
        <input
          className="color-swatch"
          type="color"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
        <input
          className="control control--color-text"
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
      </div>
    </label>
  );
}

function FeatureInput({
  label,
  value,
  max = 10,
  onChange,
}: {
  label: string;
  value: number;
  max?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="feature-input">
      <div className="feature-input__row">
        <span>{label}</span>
        <span>{value.toFixed(1)}</span>
      </div>
      <input
        className="feature-input__slider"
        type="range"
        min={0}
        max={max}
        step={0.1}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function TuneInput({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="feature-input">
      <div className="feature-input__row">
        <span>{label}</span>
        <span>{value.toFixed(step < 1 ? 2 : 1)}</span>
      </div>
      <input
        className="feature-input__slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function DetailFeatureCell({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="detail-feature-cell">
      <strong>
        {value.toFixed(1)}
        <small>/10</small>
      </strong>
      <span>{label}</span>
    </div>
  );
}

function ScoreBox({
  label,
  score,
  tier,
  gradientStart,
  gradientEnd,
  barStart,
  barEnd,
  barWidth,
  revealStyle,
}: {
  label: string;
  score: number;
  tier: string;
  gradientStart: string;
  gradientEnd: string;
  barStart: string;
  barEnd: string;
  barWidth: number;
  revealStyle?: React.CSSProperties;
}) {
  return (
    <div
      className="score-box"
      style={{
        background: `linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015) 30%, rgba(10,10,12,0.92) 68%), linear-gradient(180deg, ${gradientStart}, ${gradientEnd})`,
        borderColor: `${barStart}33`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.08), 0 22px 44px rgba(0,0,0,0.28), 0 0 40px ${barStart}22, 0 0 72px ${barStart}14`,
        ['--card-accent' as string]: barStart,
        ['--card-accent-soft' as string]: `${barEnd}22`,
        ['--card-accent-faint' as string]: `${barStart}12`,
        ...revealStyle,
      }}
    >
      <span className="score-box__label">{label}</span>
      <div className="score-box__main">
        <span className="score-box__score">{score.toFixed(1)}</span>
        <span className="score-box__tier">{tier}</span>
      </div>
      <div className="score-box__track">
        <div
          className="score-box__fill"
          style={{
            width: `${barWidth}%`,
            background: `linear-gradient(90deg, ${barStart} 0%, ${barEnd} 100%)`,
            color: barStart,
          }}
        />
      </div>
    </div>
  );
}

function windowedProgress(value: number, start: number, end: number) {
  if (value <= start) return 0;
  if (value >= end) return 1;
  return (value - start) / (end - start);
}

function easedProgress(value: number) {
  return 1 - Math.pow(1 - value, 3);
}

function getFaceStatusLabel(status: FaceStatus) {
  switch (status) {
    case 'detecting':
      return 'Detecting face...';
    case 'detected':
      return 'Face detected';
    case 'no-face':
      return 'No face detected';
    case 'error':
      return 'Face detection failed';
    default:
      return 'Upload an image to detect a face';
  }
}

function getDetectionStatusLabel(mode: DetectionMode, status: FaceStatus) {
  if (mode === 'body') {
    switch (status) {
      case 'detecting':
        return 'Detecting body...';
      case 'detected':
        return 'Body detected';
      case 'no-face':
        return 'No body detected';
      case 'error':
        return 'Body detection failed';
      default:
        return 'Upload an image to detect a body';
    }
  }

  return getFaceStatusLabel(status);
}

function drawFaceMesh({
  canvas,
  image,
  status,
  meshPoints,
  meshDrawProgress,
  meshColor,
  meshOpacity,
  imageOffsetX = 0,
  imageOffsetY = 0,
  imageZoom = 1,
}: {
  canvas: HTMLCanvasElement | null;
  image: HTMLImageElement | null;
  status: FaceStatus;
  meshPoints: MeshPoint[];
  meshDrawProgress: number;
  meshColor: string;
  meshOpacity?: number;
  imageOffsetX?: number;
  imageOffsetY?: number;
  imageZoom?: number;
}) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  if (!image || !image.complete || status !== 'detected' || meshPoints.length === 0) {
    return;
  }

  const { scale, offsetX, offsetY } = calculateCoverPlacement({
    frameWidth: width,
    frameHeight: height,
    sourceWidth: image.naturalWidth,
    sourceHeight: image.naturalHeight,
    imageOffsetX,
    imageOffsetY,
    imageZoom,
  });

  context.beginPath();
  const strokeOpacity = clampNumber(meshOpacity ?? 0.42, 0, 1);
  context.strokeStyle = hexToRgba(meshColor, strokeOpacity);
  context.lineWidth = 0.65;
  context.shadowColor = hexToRgba(meshColor, Math.min(strokeOpacity * 0.2, 0.12));
  context.shadowBlur = 2;

  const visibleEdgeCount = Math.floor(FACE_MESH_EDGES.length * meshDrawProgress);

  for (const [from, to] of FACE_MESH_EDGES.slice(0, visibleEdgeCount)) {
    const start = meshPoints[from];
    const end = meshPoints[to];

    if (!start || !end) {
      continue;
    }

    context.moveTo(start.x * scale + offsetX, start.y * scale + offsetY);
    context.lineTo(end.x * scale + offsetX, end.y * scale + offsetY);
  }

  context.stroke();
}

function calculateDetailedMetrics(meshPoints: MeshPoint[]): DetailedMetrics {
  const faceLeft = meshPoints[234] ?? meshPoints[127] ?? meshPoints[93];
  const faceRight = meshPoints[454] ?? meshPoints[356] ?? meshPoints[323];
  const chin = meshPoints[152];
  const brow = meshPoints[10] ?? meshPoints[9];
  const leftEye = averagePoint([meshPoints[33], meshPoints[133], meshPoints[159], meshPoints[145]]);
  const rightEye = averagePoint([meshPoints[263], meshPoints[362], meshPoints[386], meshPoints[374]]);
  const nose = meshPoints[1] ?? meshPoints[4];
  const mouth = averagePoint([meshPoints[13], meshPoints[14], meshPoints[78], meshPoints[308]]);
  const jawLeft = meshPoints[172] ?? meshPoints[136];
  const jawRight = meshPoints[397] ?? meshPoints[365];

  const faceWidth = distanceBetween(faceLeft, faceRight) || 1;
  const faceHeight = distanceBetween(brow, chin) || 1;
  const eyeSpacing = distanceBetween(leftEye, rightEye);
  const jawWidth = distanceBetween(jawLeft, jawRight);
  const midface = nose && mouth && chin ? (mouth.y - nose.y) / Math.max(chin.y - nose.y, 1) : 0.5;
  const symmetry = leftEye && rightEye ? 1 - Math.min(Math.abs(leftEye.y - rightEye.y) / faceHeight, 1) : 0.8;

  return {
    eyeSpacingPercent: clampNumber((eyeSpacing / faceWidth) * 100, 0, 100),
    jawWidthPercent: clampNumber((jawWidth / faceWidth) * 100, 0, 100),
    midfacePercent: clampNumber(midface * 100, 0, 100),
    symmetryPercent: clampNumber(symmetry * 100, 0, 100),
  };
}

function calculateBodyMetrics(meshPoints: MeshPoint[]): DetailedMetrics {
  const nose = meshPoints[0];
  const leftShoulder = meshPoints[5];
  const rightShoulder = meshPoints[6];
  const leftHip = meshPoints[11];
  const rightHip = meshPoints[12];
  const leftAnkle = meshPoints[15];
  const rightAnkle = meshPoints[16];

  const shoulderWidth = distanceBetween(leftShoulder, rightShoulder) || 1;
  const hipWidth = distanceBetween(leftHip, rightHip) || shoulderWidth;
  const shoulderCenter = averagePoint([leftShoulder, rightShoulder]);
  const hipCenter = averagePoint([leftHip, rightHip]);
  const ankleCenter = averagePoint([leftAnkle, rightAnkle]);
  const torsoHeight = distanceBetween(shoulderCenter, hipCenter) || 1;
  const bodyHeight = distanceBetween(nose, ankleCenter) || distanceBetween(shoulderCenter, ankleCenter) || 1;
  const shoulderTilt = leftShoulder && rightShoulder ? Math.abs(leftShoulder.y - rightShoulder.y) : 0;
  const hipTilt = leftHip && rightHip ? Math.abs(leftHip.y - rightHip.y) : 0;
  const centerDeviation =
    shoulderCenter && hipCenter && ankleCenter
      ? Math.abs((shoulderCenter.x + ankleCenter.x) / 2 - hipCenter.x)
      : 0;
  const stanceWidth = distanceBetween(leftAnkle, rightAnkle);

  return {
    eyeSpacingPercent: clampNumber((bodyHeight / shoulderWidth) * 10, 0, 100),
    jawWidthPercent: clampNumber((1 - (shoulderTilt + hipTilt) / Math.max(torsoHeight, 1)) * 100, 0, 100),
    midfacePercent: clampNumber((shoulderWidth / Math.max(hipWidth, 1)) * 65, 0, 100),
    symmetryPercent: clampNumber((1 - centerDeviation / Math.max(stanceWidth || shoulderWidth, 1)) * 100, 0, 100),
  };
}

function drawDetailScanOverlay({
  canvas,
  image,
  status,
  meshPoints,
  progress,
  overlayStyle,
}: {
  canvas: HTMLCanvasElement | null;
  image: HTMLImageElement | null;
  status: FaceStatus;
  meshPoints: MeshPoint[];
  progress: number;
  overlayStyle: DetailOverlayStyle;
}) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  if (!image || !image.complete || status !== 'detected' || meshPoints.length === 0) {
    return;
  }

  const naturalWidth = image.naturalWidth;
  const naturalHeight = image.naturalHeight;
  const scale = Math.max(width / naturalWidth, height / naturalHeight);
  const drawWidth = naturalWidth * scale;
  const drawHeight = naturalHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;
  const mappedPoints = meshPoints.map((point) => ({
    x: point.x * scale + offsetX,
    y: point.y * scale + offsetY,
  }));

  const faceLeft = mappedPoints[234] ?? mappedPoints[127] ?? mappedPoints[93];
  const faceRight = mappedPoints[454] ?? mappedPoints[356] ?? mappedPoints[323];
  const brow = mappedPoints[10] ?? mappedPoints[9];
  const chin = mappedPoints[152];

  if (!faceLeft || !faceRight || !brow || !chin) {
    return;
  }

  const networkIndexes = [33, 133, 1, 362, 263, 61, 291, 152];
  const networkConnections: Array<[number, number]> = [
    [33, 133],
    [133, 1],
    [1, 362],
    [362, 263],
    [1, 61],
    [1, 291],
    [61, 152],
    [291, 152],
    [61, 291],
  ];
  const visibleDots = Math.max(2, Math.round(progress * networkIndexes.length));
  const visibleLines = Math.max(1, Math.round(progress * networkConnections.length));
  const lineColor = overlayStyle.color;
  const glowColor = overlayStyle.color;

  context.strokeStyle = hexToRgba(lineColor, overlayStyle.opacity);
  context.lineWidth = overlayStyle.lineThickness;
  context.shadowColor = hexToRgba(glowColor, Math.min(overlayStyle.opacity * 0.7, 0.75));
  context.shadowBlur = 12;
  context.globalAlpha = 1;
  context.beginPath();

  for (const [fromIndex, toIndex] of networkConnections.slice(0, visibleLines)) {
    const from = mappedPoints[fromIndex];
    const to = mappedPoints[toIndex];

    if (!from || !to) {
      continue;
    }

    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
  }

  context.stroke();

  context.fillStyle = hexToRgba(glowColor, overlayStyle.opacity);
  for (const index of networkIndexes.slice(0, visibleDots)) {
    const point = mappedPoints[index];
    if (!point) {
      continue;
    }

    context.beginPath();
    context.arc(point.x, point.y, overlayStyle.dotSize, 0, Math.PI * 2);
    context.fill();
  }

  context.shadowBlur = 0;
  context.globalAlpha = 1;
}

function drawBodyScanOverlay({
  canvas,
  image,
  status,
  meshPoints,
  progress,
  overlayStyle,
}: {
  canvas: HTMLCanvasElement | null;
  image: HTMLImageElement | null;
  status: FaceStatus;
  meshPoints: MeshPoint[];
  progress: number;
  overlayStyle: DetailOverlayStyle;
}) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  if (!image || !image.complete || status !== 'detected' || meshPoints.length === 0) {
    return;
  }

  const naturalWidth = image.naturalWidth;
  const naturalHeight = image.naturalHeight;
  const scale = Math.max(width / naturalWidth, height / naturalHeight);
  const drawWidth = naturalWidth * scale;
  const drawHeight = naturalHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;
  const mappedPoints = meshPoints.map((point) => ({
    x: point.x * scale + offsetX,
    y: point.y * scale + offsetY,
  }));

  const bodyKeyIndexes = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
  const visibleDots = Math.max(2, Math.round(progress * bodyKeyIndexes.length));
  const visibleLines = Math.max(1, Math.round(progress * BODY_CONNECTIONS.length));

  context.strokeStyle = hexToRgba(overlayStyle.color, overlayStyle.opacity);
  context.lineWidth = overlayStyle.lineThickness;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.shadowColor = hexToRgba(overlayStyle.color, Math.min(overlayStyle.opacity * 0.72, 0.78));
  context.shadowBlur = 12;
  context.beginPath();

  for (const [fromIndex, toIndex] of BODY_CONNECTIONS.slice(0, visibleLines)) {
    const from = mappedPoints[fromIndex];
    const to = mappedPoints[toIndex];

    if (!from || !to) {
      continue;
    }

    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
  }

  context.stroke();

  context.fillStyle = hexToRgba(overlayStyle.color, overlayStyle.opacity);
  for (const index of bodyKeyIndexes.slice(0, visibleDots)) {
    const point = mappedPoints[index];
    if (!point) {
      continue;
    }

    context.beginPath();
    context.arc(point.x, point.y, overlayStyle.dotSize, 0, Math.PI * 2);
    context.fill();
  }

  context.shadowBlur = 0;
  context.globalAlpha = 1;
}

function drawSideProfileOverlay({
  canvas,
  image,
  status,
  meshPoints,
  progress,
  overlayStyle,
}: {
  canvas: HTMLCanvasElement | null;
  image: HTMLImageElement | null;
  status: FaceStatus;
  meshPoints: MeshPoint[];
  progress: number;
  overlayStyle: DetailOverlayStyle;
}) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  if (!image || !image.complete || status !== 'detected' || meshPoints.length === 0) {
    return;
  }

  const naturalWidth = image.naturalWidth;
  const naturalHeight = image.naturalHeight;
  const scale = Math.max(width / naturalWidth, height / naturalHeight);
  const drawWidth = naturalWidth * scale;
  const drawHeight = naturalHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;
  const mappedPoints = meshPoints.map((point) => ({
    x: point.x * scale + offsetX,
    y: point.y * scale + offsetY,
  }));

  const faceLeft = mappedPoints[234] ?? mappedPoints[127] ?? mappedPoints[93];
  const faceRight = mappedPoints[454] ?? mappedPoints[356] ?? mappedPoints[323];

  if (!faceLeft || !faceRight) {
    return;
  }

  const faceCenterX = (faceLeft.x + faceRight.x) / 2;
  const rightJawPathIndexes = [454, 323, 361, 397, 365, 379, 378, 400, 377, 152];
  const leftJawPathIndexes = [234, 93, 132, 172, 136, 150, 149, 176, 148, 152];

  const scoreContour = (indexes: number[]) => {
    const points = indexes
      .map((index) => mappedPoints[index])
      .filter((point): point is MeshPoint => Boolean(point));

    if (points.length === 0) {
      return -1;
    }

    return points.reduce((sum, point) => sum + Math.abs(point.x - faceCenterX), 0) / points.length;
  };

  const useRightContour = scoreContour(rightJawPathIndexes) >= scoreContour(leftJawPathIndexes);
  const jawPathIndexes = useRightContour ? rightJawPathIndexes : leftJawPathIndexes;
  const zygomaPoint = useRightContour ? mappedPoints[323] ?? mappedPoints[454] : mappedPoints[93] ?? mappedPoints[234];
  const ramusTop = useRightContour ? mappedPoints[361] ?? mappedPoints[323] : mappedPoints[132] ?? mappedPoints[93];
  const jawAnglePoint = useRightContour ? mappedPoints[397] ?? mappedPoints[365] : mappedPoints[172] ?? mappedPoints[136];
  const visibleDots = Math.max(3, Math.round(progress * Math.min(jawPathIndexes.length, 6)));
  const visibleLinePoints = Math.max(3, Math.round(progress * jawPathIndexes.length));
  const jawLine = jawPathIndexes
    .slice(0, visibleLinePoints)
    .map((index) => mappedPoints[index])
    .filter((point): point is MeshPoint => Boolean(point));
  const dotIndexes = useRightContour ? [323, 361, 397, 365, 152] : [93, 132, 172, 136, 152];

  context.strokeStyle = hexToRgba(overlayStyle.color, overlayStyle.opacity);
  context.lineWidth = Math.max(1, overlayStyle.lineThickness * 0.95);
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.shadowColor = hexToRgba(overlayStyle.color, Math.min(overlayStyle.opacity * 0.72, 0.78));
  context.shadowBlur = 10;

  if (jawLine.length >= 2) {
    context.beginPath();
    context.moveTo(jawLine[0].x, jawLine[0].y);
    for (const point of jawLine.slice(1)) {
      context.lineTo(point.x, point.y);
    }
    context.stroke();
  }

  if (zygomaPoint && ramusTop && progress > 0.12) {
    const zygomaProgress = windowedProgress(progress, 0.12, 0.52);
    const endX = zygomaPoint.x + (ramusTop.x - zygomaPoint.x) * zygomaProgress;
    const endY = zygomaPoint.y + (ramusTop.y - zygomaPoint.y) * zygomaProgress;
    context.globalAlpha = 0.88;
    context.beginPath();
    context.moveTo(zygomaPoint.x, zygomaPoint.y);
    context.lineTo(endX, endY);
    context.stroke();
    context.globalAlpha = 1;
  }

  if (ramusTop && jawAnglePoint && progress > 0.2) {
    const ramusProgress = windowedProgress(progress, 0.2, 0.8);
    const endX = ramusTop.x + (jawAnglePoint.x - ramusTop.x) * ramusProgress;
    const endY = ramusTop.y + (jawAnglePoint.y - ramusTop.y) * ramusProgress;
    context.globalAlpha = 0.9;
    context.beginPath();
    context.moveTo(ramusTop.x, ramusTop.y);
    context.lineTo(endX, endY);
    context.stroke();
    context.globalAlpha = 1;
  }

  context.fillStyle = hexToRgba(overlayStyle.color, overlayStyle.opacity);
  for (const index of dotIndexes.slice(0, visibleDots)) {
    const point = mappedPoints[index];
    if (!point) {
      continue;
    }

    context.beginPath();
    context.arc(point.x, point.y, Math.max(overlayStyle.dotSize * 0.9, 2), 0, Math.PI * 2);
    context.fill();
  }

  context.shadowBlur = 0;
  context.globalAlpha = 1;
}

function createSyntheticSideProfilePoints(box: {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
}): MeshPoint[] {
  const points: MeshPoint[] = [];
  const x = box.xMin;
  const y = box.yMin;
  const width = box.width;
  const height = box.height;

  const setPoint = (index: number, px: number, py: number) => {
    points[index] = { x: px, y: py };
  };

  setPoint(454, x + width * 0.78, y + height * 0.24);
  setPoint(323, x + width * 0.74, y + height * 0.34);
  setPoint(361, x + width * 0.75, y + height * 0.45);
  setPoint(397, x + width * 0.71, y + height * 0.58);
  setPoint(365, x + width * 0.67, y + height * 0.7);
  setPoint(379, x + width * 0.63, y + height * 0.79);
  setPoint(378, x + width * 0.58, y + height * 0.86);
  setPoint(400, x + width * 0.52, y + height * 0.92);
  setPoint(377, x + width * 0.46, y + height * 0.97);
  setPoint(152, x + width * 0.41, y + height * 1.0);

  setPoint(234, x + width * 0.22, y + height * 0.24);
  setPoint(93, x + width * 0.26, y + height * 0.34);
  setPoint(132, x + width * 0.25, y + height * 0.45);
  setPoint(172, x + width * 0.29, y + height * 0.58);
  setPoint(136, x + width * 0.33, y + height * 0.7);
  setPoint(150, x + width * 0.37, y + height * 0.79);
  setPoint(149, x + width * 0.42, y + height * 0.86);
  setPoint(176, x + width * 0.48, y + height * 0.92);
  setPoint(148, x + width * 0.54, y + height * 0.97);

  return points;
}

function averagePoint(points: Array<MeshPoint | undefined>) {
  const filtered = points.filter((point): point is MeshPoint => Boolean(point));
  if (filtered.length === 0) {
    return undefined;
  }

  const sum = filtered.reduce(
    (accumulator, point) => ({
      x: accumulator.x + point.x,
      y: accumulator.y + point.y,
    }),
    { x: 0, y: 0 },
  );

  return {
    x: sum.x / filtered.length,
    y: sum.y / filtered.length,
  };
}

function distanceBetween(a?: MeshPoint, b?: MeshPoint) {
  if (!a || !b) {
    return 0;
  }

  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function hexToRgba(hex: string, alpha: number) {
  const normalized = hex.trim().replace('#', '');
  const safeAlpha = clampNumber(alpha, 0, 1);

  if (!/^[0-9a-fA-F]{3}$|^[0-9a-fA-F]{6}$/.test(normalized)) {
    return `rgba(255, 90, 82, ${safeAlpha})`;
  }

  const expanded = normalized.length === 3
    ? normalized
        .split('')
        .map((char) => char + char)
        .join('')
    : normalized;

  const red = Number.parseInt(expanded.slice(0, 2), 16);
  const green = Number.parseInt(expanded.slice(2, 4), 16);
  const blue = Number.parseInt(expanded.slice(4, 6), 16);

  return `rgba(${red}, ${green}, ${blue}, ${safeAlpha})`;
}

function calculateCoverPlacement({
  frameWidth,
  frameHeight,
  sourceWidth,
  sourceHeight,
  imageOffsetX = 0,
  imageOffsetY = 0,
  imageZoom = 1,
}: {
  frameWidth: number;
  frameHeight: number;
  sourceWidth: number;
  sourceHeight: number;
  imageOffsetX?: number;
  imageOffsetY?: number;
  imageZoom?: number;
}) {
  const safeSourceWidth = Math.max(sourceWidth, 1);
  const safeSourceHeight = Math.max(sourceHeight, 1);
  const scale = Math.max(frameWidth / safeSourceWidth, frameHeight / safeSourceHeight);
  const drawWidth = safeSourceWidth * scale;
  const drawHeight = safeSourceHeight * scale;
  const anchorX = (50 + imageOffsetX) / 100;
  const anchorY = (50 + imageOffsetY) / 100;
  const zoom = Math.max(imageZoom, 1);
  const zoomedWidth = drawWidth * zoom;
  const zoomedHeight = drawHeight * zoom;
  const baseOffsetX = (frameWidth - drawWidth) * anchorX;
  const baseOffsetY = (frameHeight - drawHeight) * anchorY;

  return {
    scale: scale * zoom,
    drawWidth: zoomedWidth,
    drawHeight: zoomedHeight,
    offsetX: baseOffsetX - (zoomedWidth - drawWidth) / 2,
    offsetY: baseOffsetY - (zoomedHeight - drawHeight) / 2,
  };
}

async function createCroppedAvatarImage({
  source,
  offsetX,
  offsetY,
  zoom,
  size,
}: {
  source: string;
  offsetX: number;
  offsetY: number;
  zoom: number;
  size: number;
}) {
  const image = await loadImageElement(source);
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Could not create image crop.');
  }

  const placement = calculateCoverPlacement({
    frameWidth: size,
    frameHeight: size,
    sourceWidth: image.naturalWidth,
    sourceHeight: image.naturalHeight,
    imageOffsetX: offsetX,
    imageOffsetY: offsetY,
    imageZoom: zoom,
  });

  context.clearRect(0, 0, size, size);
  context.drawImage(image, placement.offsetX, placement.offsetY, placement.drawWidth, placement.drawHeight);
  return canvas.toDataURL('image/png');
}

async function loadImageElement(source: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not load image.'));
    image.src = source;
  });
}

function CropSlider({
  label,
  valueLabel,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  valueLabel: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="crop-slider">
      <div className="crop-slider__row">
        <span>{label}</span>
        <strong>{valueLabel}</strong>
      </div>
      <input
        className="crop-slider__input"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function waitForAnimationFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
}

async function waitForExportNodeReady(node: HTMLElement) {
  const images = Array.from(node.querySelectorAll('img'));

  await Promise.all(
    images.map((image) => {
      if (image.complete && image.naturalWidth > 0) {
        return typeof image.decode === 'function'
          ? image.decode().catch(() => undefined)
          : Promise.resolve();
      }

      return new Promise<void>((resolve) => {
        const handleDone = () => {
          image.removeEventListener('load', handleDone);
          image.removeEventListener('error', handleDone);
          resolve();
        };

        image.addEventListener('load', handleDone, { once: true });
        image.addEventListener('error', handleDone, { once: true });
      });
    }),
  );

  await waitForAnimationFrame();
  await waitForAnimationFrame();
  await wait(60);
}

function renderExportCanvas({
  node,
  width,
  height,
  canvasWidth,
  canvasHeight,
}: {
  node: HTMLElement;
  width: number;
  height: number;
  canvasWidth: number;
  canvasHeight: number;
}) {
  return toCanvas(node, {
    cacheBust: true,
    width,
    height,
    canvasWidth,
    canvasHeight,
    pixelRatio: 1,
    backgroundColor: '#000000',
    skipAutoScale: true,
    style: {
      width: `${width}px`,
      height: `${height}px`,
    },
    filter: (node) => !(node instanceof HTMLElement && node.dataset.exportIgnore === 'true'),
  });
}

function upscaleCanvas(source: HTMLCanvasElement, width: number, height: number) {
  if (source.width === width && source.height === height) {
    return source;
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');

  if (!context) {
    throw new Error('Could not create the export canvas.');
  }

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, 0, 0, width, height);
  return canvas;
}

function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }

      reject(new Error('Could not create a PNG frame.'));
    }, 'image/png');
  });
}

function describeExportError(error: unknown) {
  if (error instanceof Event) {
    const target = error.target;
    if (target instanceof HTMLImageElement) {
      return 'Could not load one of the preview images during export. Try re-uploading the avatar as PNG or JPG.';
    }

    return 'MP4 export failed while rendering the preview.';
  }

  if (error instanceof Error) {
    return error.message || 'MP4 export failed.';
  }

  if (typeof error === 'string' && error.trim()) {
    return error;
  }

  if (error && typeof error === 'object') {
    const entries = Object.entries(error as Record<string, unknown>)
      .map(([key, value]) => `${key}: ${String(value)}`)
      .join(', ');

    if (entries) {
      return entries;
    }
  }

  return `MP4 export failed (${String(error)})`;
}

function getRevealStyle(progress: number, translateY: number, startScale: number): React.CSSProperties {
  return {
    opacity: progress,
    transform: `translateY(${(1 - progress) * translateY}px) scale(${startScale + (1 - startScale) * progress})`,
  };
}

export default App;
