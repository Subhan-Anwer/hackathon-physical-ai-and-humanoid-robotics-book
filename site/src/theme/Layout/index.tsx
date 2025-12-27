import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import Chatbot from '@site/src/components/Chatbot';

type LayoutProps = {
  children: React.ReactNode;
};

export default function Layout(props: LayoutProps): JSX.Element {
  return (
    <>
      <OriginalLayout {...props} />
      <Chatbot />
    </>
  );
}