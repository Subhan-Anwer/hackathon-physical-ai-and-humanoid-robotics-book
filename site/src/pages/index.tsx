import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

// Icons as simple SVG components
const RosIcon = () => (
  <svg className={styles.bookIcon} viewBox="0 0 24 24" width="48" height="48">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" fill="none"/>
  </svg>
);

const SimulationIcon = () => (
  <svg className={styles.bookIcon} viewBox="0 0 24 24" width="48" height="48">
    <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2" fill="none"/>
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" fill="none"/>
  </svg>
);

const AiIcon = () => (
  <svg className={styles.bookIcon} viewBox="0 0 24 24" width="48" height="48">
    <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M2 17l10 5 10-5" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" fill="none"/>
  </svg>
);

const VlaIcon = () => (
  <svg className={styles.bookIcon} viewBox="0 0 24 24" width="48" height="48">
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" stroke="currentColor" strokeWidth="2" fill="none"/>
    <polyline points="7.5,12 12,15 16.5,12" stroke="currentColor" strokeWidth="2"/>
  </svg>
);

const BookStructureIcon = () => (
  <svg className={styles.bookIcon} viewBox="0 0 24 24" width="24" height="24">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" stroke="currentColor" strokeWidth="2" fill="none"/>
  </svg>
);

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <Heading as="h1" className="hero__title">
            Physical AI & Humanoid Robotics
          </Heading>
          <p className="hero__subtitle">
            A practical, hands-on guide to building intelligent robots in the
            physical world
          </p>
          <p className={styles.heroDescription}>
            Learn to create embodied intelligence through ROS 2, simulation, AI,
            and Vision-Language-Action systems. This comprehensive resource
            covers everything from robotic foundations to advanced AI
            integration.
          </p>
          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro"
            >
              Start Reading
            </Link>
            <Link
              className="button button--secondary button--lg"
              to="https://github.com/subhan-anwer/hackathon-physical-ai-and-humanoid-robotics-book/"
              target="_blank"
            >
              View on GitHub
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function WhatYouWillLearn() {
  const learnItems = [
    {
      title: 'Robotic Nervous System (ROS 2)',
      description: 'Master the Robot Operating System as the foundation for robot communication, control, and coordination.',
      icon: <RosIcon />
    },
    {
      title: 'Digital Twins (Simulation)',
      description: 'Create and work with realistic simulations using Gazebo and Unity for safe robot development.',
      icon: <SimulationIcon />
    },
    {
      title: 'AI & Decision Making (NVIDIA Isaac)',
      description: 'Implement intelligent robot behavior using NVIDIA Isaac for perception, planning, and control.',
      icon: <AiIcon />
    },
    {
      title: 'Vision-Language-Action & Capstone',
      description: 'Build advanced systems that combine vision, language understanding, and physical action.',
      icon: <VlaIcon />
    }
  ];

  return (
    <section className={clsx(styles.section, styles.grayBackground)}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" className={clsx('margin-bottom--lg', styles.sectionTitle)}>
              What You Will Learn
            </Heading>
          </div>
        </div>
        <div className="row">
          {learnItems.map((item, index) => (
            <div key={index} className="col col--3 margin-bottom--lg">
              <div className={styles.learnCard}>
                <div className={styles.iconContainer}>
                  {item.icon}
                </div>
                <h3 className={styles.learnCardTitle}>{item.title}</h3>
                <p className={styles.learnCardDescription}>{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function BookStructure() {
  const modules = [
    { title: 'Introduction', description: 'Foundations of Physical AI and Robotics' },
    { title: 'Module 1: ROS 2', description: 'The Robotic Nervous System' },
    { title: 'Module 2: Gazebo & Unity', description: 'Digital Twins and Simulation' },
    { title: 'Module 3: NVIDIA Isaac', description: 'The AI Robot Brain' },
    { title: 'Module 4: Vision-Language-Action', description: 'Advanced Perception and Action' },
    { title: 'Capstone Project', description: 'Complete Robot Implementation' }
  ];

  return (
    <section className={styles.section}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" className={clsx('margin-bottom--lg', styles.sectionTitle)}>
              Book Structure
            </Heading>
          </div>
        </div>
        <div className="row">
          {modules.map((module, index) => (
            <div key={index} className="col col--4 margin-bottom--lg">
              <div className={styles.moduleCard}>
                <div className={styles.moduleIcon}>
                  <BookStructureIcon />
                </div>
                <h3 className={styles.moduleTitle}>{module.title}</h3>
                <p className={styles.moduleDescription}>{module.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function HandsOnFocus() {
  return (
    <section className={clsx(styles.section, styles.grayBackground)}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <div className={styles.handsOnContainer}>
              <Heading as="h2" className={clsx('margin-bottom--lg', styles.sectionTitle)}>
                Hands-On Learning Approach
              </Heading>
              <p className={styles.handsOnDescription}>
                This book emphasizes practical implementation with hands-on labs, simulation exercises,
                and real-world robotics projects. Each module includes guided exercises that build
                upon the theoretical concepts to create actual working robot systems.
              </p>
              <ul className={styles.handsOnList}>
                <li>Interactive labs with step-by-step instructions</li>
                <li>Simulation environments for safe experimentation</li>
                <li>Real-world mindset with practical applications</li>
                <li>Capstone project integrating all concepts</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function FinalCTA() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <div className={styles.finalCTA}>
              <Heading as="h2" className={clsx('margin-bottom--lg', styles.sectionTitle)}>
                Begin the Journey
              </Heading>
              <p className={styles.finalCTADescription}>
                Start building intelligent robots that interact with the physical world.
                Master the essential technologies and concepts that power modern robotics.
              </p>
              <div className={styles.finalButtons}>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/module-1-ros2/chapter-1">
                  Start Module 1
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="A practical, hands-on guide to building intelligent robots in the physical world">
      <HomepageHeader />
      <main>
        <WhatYouWillLearn />
        <BookStructure />
        <HandsOnFocus />
        <FinalCTA />
      </main>
    </Layout>
  );
}
