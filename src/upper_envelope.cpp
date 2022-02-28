// ALGORITHM AND IMPLEMENTATION PROPERTY OF RINAT ABDRASHITOV
#include <igl/nchoosek.h>
#include <igl/mat_max.h>
#include <igl/slice_mask.h>
#include <igl/edges.h>
#include <assert.h>
#include <limits>
#include <igl/boundary_facets.h>
#include <igl/extract_manifold_patches.h>
#include <igl/remove_unreferenced.h>
#include <igl/is_vertex_manifold.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <ctime>

typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ArrayXb;

void boundary_is_manifold(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T) {
    Eigen::MatrixXi F;
    igl::boundary_facets(T, F);
    std::cout << "boundary size" << std::endl;

    Eigen::MatrixXi C;
    igl::extract_manifold_patches(F, C);
    Eigen::MatrixXd RV;
    Eigen::MatrixXi tmp1, tmp2, IM;
    igl::remove_unreferenced(V, F, RV, tmp1, IM, tmp2);
}

void carve_out_material(Eigen::MatrixXi &FT, int &idxF, ArrayXb &LT, const Eigen::MatrixXd& DT, int mid, double material_threshold, Eigen::MatrixXi &CFT) {

    ArrayXb CT(FT.rows(), 1);
    CT.setConstant(false);

    for (int i = 0; i < idxF-1; i++) {
        Eigen::MatrixXd D;
	    igl::slice(DT, FT.row(i).transpose(), 1, D);

	    Eigen::VectorXi I(D.rows());
        Eigen::VectorXd tmp;
	    igl::mat_max(D, 2, tmp, I);

	if ((I.array()==I(0)).all() && I(0) == mid) {
            CT(i) = true;
	}
	else {
        ArrayXb LL(1, D.cols());
	    LL.setConstant(false);
	    LL(mid) = true;
	    Eigen::MatrixXd d = D;
	    d.colwise() -= D.col(mid);

	    for (int j = 0; j < D.cols(); j++) {
                if (j != mid) {
                    Eigen::VectorXd t = d.col(j);
		    if ((t.array() <= 0).all()) {
                        LL(j) = true;
		    }
		    else {
		        Eigen::VectorXd a = igl::slice_mask(t, t.array() > 0, 1);
			if ((a.array().abs() < material_threshold).all()) {
                            LL(j) = true;
			}
		    }
		}
	    }

	    CT(i) = LL.array().all();
	}
    }

    CFT = igl::slice_mask(FT, CT, 1);
    LT = igl::slice_mask(LT, !CT, 1);
    FT = igl::slice_mask(FT, !CT, 1);

    for (int iii = 0; iii < FT.rows(); iii++) {
        if (FT(iii,0) == 0 && FT(iii,1) == 0 && FT(iii,2) == 0 && FT(iii,3) == 0) 
	{
	  idxF = iii;
	  break;
	}
    }

    //std::cout << idxF << std::endl;

}

bool only_one_material(const ArrayXb& LT) {
    Eigen::VectorXi s = LT.cast<int>().rowwise().sum();
    return !(s.array() > 1).any();
}

bool isclose(double a, double b, double abs_tol=1e-9) {
    return std::abs(a-b) <= abs_tol;
}

void remove_material_quick(const Eigen::MatrixXd& D, const Eigen::MatrixXi& F, 
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> &L, int tetid, bool &removed) {
  // quickly check that we only have one material
  if (L.row(tetid).cast<int>().sum() == 1) return;  // sum for booleans is always 0/1

  Eigen::VectorXi I(4);
  D.row(F(tetid,0)).maxCoeff(&I(0));
  D.row(F(tetid,1)).maxCoeff(&I(1));
  D.row(F(tetid,2)).maxCoeff(&I(2));
  D.row(F(tetid,3)).maxCoeff(&I(3)); 
//  Eigen::MatrixXd B(4, D.cols());
//  Eigen::VectorXi tmp, I;
//  igl::slice(D, F.row(tetid).transpose(), 1, B);
//  igl::mat_max(B, 2, tmp, I);

  removed = false;
  if ((I.array()==I(0)).all()) {
    L.row(tetid).setConstant(false);
    L(tetid, I(0)) = true;
    removed = true;
  }

}

void remove_unused_material2(const Eigen::MatrixXd& DT, const Eigen::MatrixXi& FT, Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> &LT, int tetid,
    double threshold, bool &deleted) {
    Eigen::MatrixXd D;
    igl::slice(DT, FT.row(tetid).transpose(), 1, D);
//    std::cout << DT << std::endl;
//    std::cout << FT.row(tetid) << std::endl;
//    std::cout << D << std::endl;
    ArrayXb L = LT.row(tetid);
    ArrayXb l(1, D.cols());
    l.setConstant(false);

    for (int i = 0; i < D.cols(); i++) {
        Eigen::MatrixXd d = D;
	d.colwise() -= D.col(i);
	for (int j = 0; j < D.cols(); j++) {
            if ((j != i) && L(j)) {
                Eigen::VectorXd t = d.col(j);
                if ((t.array()>=0).all()) l(i) = true;
		else {
                    Eigen::VectorXd a = igl::slice_mask(t, t.array() < 0, 1); 

		    if ((a.array().abs() < threshold).all()) l(i) = true;
		}
	    }
	}
    }

    deleted = l.array().any();
    bool allequal = l.array().all();

    if (allequal) {
       l.setConstant(1, D.cols(), true);
       l(0) = false;
    }

    ArrayXb tmp;
    tmp.setConstant(1, l.cols(), false);
    l.select(tmp, LT.row(tetid));

}


void propogate_split_func(Eigen::MatrixXd &inV, int &idxV, Eigen::MatrixXi &inF, int &idxF,const Eigen::VectorXd& inD, Eigen::MatrixXd &allD, ArrayXb &inL, Eigen::SparseMatrix<double> &uE2V,
    int tetid, const Eigen::Vector2i& mpair, const double iso, const double threshold) {


    auto tet_contains_edge = [&](const Eigen::RowVectorXi& t, const Eigen::RowVectorXi& e) {
        return (t.array() == e(0)).any() && (t.array() == e(1)).any();
    };

    auto VertexInterp = [&](double iso, const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2, 
	double valp1, double valp2, Eigen::RowVectorXd &p, double &mu, double threshold=1e-9) {

      if (isclose(iso, valp1, threshold)) {
          p = p1;
	  return;
      }

      if (isclose(iso, valp2, threshold)) {
          p = p2;
	  return;
      }

      if (isclose(valp1, valp2, threshold)) {
	  p = p1;
	  assert(false);
	  return;
      }

      mu = (iso - valp1) / (valp2 - valp1);
      p = p1 + mu * (p2 - p1);
        
      
    };
    

    auto split_2_edge = [&](const Eigen::RowVectorXi& t, const Eigen::RowVectorXi& e, 
	double vid, Eigen::MatrixXi &tt) {

      assert(t.cols() == 4 && e.cols() == 2);
      Eigen::RowVectorXi op = igl::slice_mask(t, !(t.array() == e(0) || t.array() == e(1)), 2);

      tt.resize(2,4);
      tt(0,0) = e(0); tt(0,1) = vid; tt(0,2) = op(0); tt(0,3) = op(1);
      tt(1,0) = e(1); tt(1,1) = vid; tt(1,2) = op(0); tt(1,3) = op(1);

    };

    auto split_2_edge_array = [&](const Eigen::RowVectorXi& t, const Eigen::MatrixXi& E, 
	const Eigen::VectorXd& V, Eigen::MatrixXi &tt) {

        assert(t.cols() == 4);
	assert(V.rows() == E.rows());

	Eigen::MatrixXi newt;
	tt.resize(1, t.cols());
	tt.row(0) = t;

	for (int i = 0; i < E.rows(); i++) {
          Eigen::RowVectorXi e = E.row(i);
	  double vid = V(i);
	  for (int j = 0; j < tt.rows(); j++) {
            Eigen::RowVectorXi curt = tt.row(j);
	    if (tet_contains_edge(curt, e)) {
	      Eigen::MatrixXi st;
              split_2_edge(curt, e, vid, st);
	      int newt_r_old = newt.rows();
	      newt.conservativeResize(newt.rows()+st.rows(), st.cols());
	      newt.block(newt_r_old, 0, st.rows(), st.cols()) = st;
	    }
	    else {
              int newt_r_old = newt.rows();
	      newt.conservativeResize(newt.rows()+curt.rows(), curt.cols());
	      newt.block(newt_r_old, 0, curt.rows(), curt.cols()) = curt;
	    }
	  }
	  tt.resize(newt.rows(), newt.cols());
	  tt = newt;
	  newt.resize(0,0);
	}
      
    };


    Eigen::RowVectorXi F = inF.row(tetid);

    Eigen::MatrixXd inD_F;
    igl::slice(inD, F.transpose(), 1, inD_F);
    if ((inD_F.array() >= iso).all() || (inD_F.array() <= iso).all()) return;

    Eigen::MatrixXi EF_unsorted(6,2), EF(6,2);
    EF_unsorted << F(0), F(1),
                   F(0), F(2),
	           F(0), F(3),
	           F(1), F(2),
	           F(1), F(3),
	           F(2), F(3);
    igl::sort(EF_unsorted, 2, true, EF);

    Eigen::MatrixXi E;
    for (int ii = 0; ii < EF.rows(); ii++) {
        double d1 = inD(EF(ii,0));
	double d2 = inD(EF(ii,1));

        if ((d1 < iso && d2 > iso) || (d1 > iso && d2 < iso)) {
            if (!isclose(d1, iso, threshold) && !isclose(d2, iso, threshold)) {
	       int r_E = E.rows();
               E.conservativeResize(r_E+1, 2);
	       E.row(r_E) = EF.row(ii);
	    }
	}
    }

    int ss = E.rows();
    if (ss == 0) return;

    assert(ss > 0);
    Eigen::MatrixXi E_tmp = E;
    igl::sortrows(E_tmp, true, E);
    Eigen::VectorXd vids = Eigen::MatrixXd::Zero(ss,1);

    //std::cout << E << std::endl;

    for (int ii = 0; ii < ss; ii++) {
        double uE2V_val = uE2V.coeffRef(E(ii,0), E(ii,1));
        assert(uE2V_val != 0);
		if (uE2V_val > 0) {
			vids(ii) = uE2V_val;
		}
		else {
			Eigen::RowVectorXd p;
			double mu;
				VertexInterp(iso, inV.row(E(ii,0)), inV.row(E(ii,1)), inD(E(ii,0)), inD(E(ii,1)), p, mu);
				inV.row(idxV) = p;
			vids(ii) = idxV;

			uE2V.coeffRef(E(ii,0), E(ii,1)) = vids(ii);
			Eigen::RowVectorXd BB = Eigen::MatrixXd::Zero(1,4);
				for (int ibb = 0; ibb < 4; ibb++) {
					if (F(ibb) == E(ii,0)) BB(ibb) = 1-mu;
			if (F(ibb) == E(ii,1)) BB(ibb) = mu;
			}

			Eigen::MatrixXd allD_slice;
			igl::slice(allD, F.transpose(), 1, allD_slice);
			allD.row(idxV) = BB * allD_slice;

			idxV++;

		}

    }


    Eigen::MatrixXi tt;
    split_2_edge_array(F, E, vids, tt);

    Eigen::RowVectorXi appendFids = Eigen::RowVectorXi::LinSpaced(tt.rows()-1, idxF, idxF+tt.rows()-2);
    Eigen::RowVectorXi aa(appendFids.cols()+1);
    aa << tetid, appendFids;
    inF.row(tetid) = tt.row(0);
    inF.block(idxF, 0, tt.rows()-1, inF.cols()) = tt.block(1, 0, tt.rows()-1, tt.cols());
    //std::cout << "inF block" << std::endl << inF.block(idxF, 0, tt.rows()-1, inF.cols()) << std::endl;

    inL.block(idxF, 0, tt.rows()-1, inL.cols()) = inL.row(tetid).replicate(tt.rows()-1,1);
    //std::cout << "inL block" << std::endl << inL.block(idxF, 0, tt.rows()-1, inL.cols());

    idxF = idxF + tt.rows() - 1;

}

void upper_envelope(Eigen::MatrixXd & VT, Eigen::MatrixXi & FT,  Eigen::MatrixXd & DT, Eigen::MatrixXd & UT, Eigen::MatrixXi & GT, ArrayXb & LT)
{

  LT.setConstant(FT.rows(), DT.cols(), true);


  double iso = 0;
  double material_threshold = 1e-5;
  Eigen::MatrixXi material_pairs_all;
  igl::nchoosek(Eigen::VectorXi::LinSpaced(DT.cols(), 0, DT.cols()-1), 2, material_pairs_all);

  //start = clock();
  int Fsize = FT.rows();
  for (int i = 0; i < Fsize; i++) {
      bool removed;
      remove_material_quick(DT, FT, LT, i, removed);
      if (!removed) {
	  bool deleted;
          remove_unused_material2(DT, FT, LT, i, material_threshold, deleted);
      }
  }
  //end = clock();
  //std::cout << "Quick remove unused materials takes: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << std::endl;

  assert(LT.cast<int>().rowwise().sum().cast<bool>().count() == LT.rows());

  int idxV = VT.rows(),
      idxF = FT.rows(),
      Vexpandsize = VT.rows(),
      Fexpandsize = FT.rows();
  int idxCFT = 1;
  Eigen::MatrixXi CFTarray(2*FT.rows(), FT.cols());
  CFTarray.setConstant(0);
  ArrayXb CLTarray(2*FT.rows(), LT.cols());
  CLTarray.setConstant(false);

  for (int mid = 0; mid < DT.cols(); mid++) {
    Eigen::MatrixXi material_pairs = igl::slice_mask(material_pairs_all, material_pairs_all.col(0).array() == mid, 1);

    ArrayXb LLT = igl::slice_mask(LT, LT.col(mid).array(), 1).cast<int>().colwise().sum() > 0;

    //start = clock();
    for (int ii = 0; ii < material_pairs.rows(); ii++) {
	Eigen::RowVectorXi mpair = material_pairs.row(ii);

	if (LLT(mpair(0)) && LLT(mpair(1))) {
            int Fsize = idxF,
		Vsize = idxV;

	    Eigen::MatrixXi uE;
	    igl::edges(FT.block(0,0,Fsize,FT.cols()), uE);
	    //std::cout << uE << std::endl;

	    typedef Eigen::Triplet<double> T;
	    std::vector<T> tripletList;
	    tripletList.reserve(uE.rows());
	    for(int iii = 0; iii < uE.rows(); iii++) {
	        tripletList.push_back(T(uE(iii,0), uE(iii,1), -1));
	    }
	    Eigen::SparseMatrix<double> uE2V(uE.col(0).maxCoeff()+1, uE.col(1).maxCoeff()+1);
	    uE2V.setFromTriplets(tripletList.begin(), tripletList.end());
            
	    Eigen::VectorXd inD = DT.block(0,mpair(0),Vsize,1) - DT.block(0,mpair(1),Vsize,1);

	    for (int i = 0; i < Fsize; i++) {
                if (idxV + 10 + 1 > VT.rows()) {
		    int idv = VT.rows();
		    int idd = DT.rows();
                    VT.conservativeResize(VT.rows()+Vexpandsize, VT.cols());
		    VT.block(idv, 0, Vexpandsize, VT.cols()).setConstant(0);
		    DT.conservativeResize(DT.rows()+Vexpandsize, DT.cols());
		    DT.block(idd, 0, Vexpandsize, DT.cols()).setConstant(0);
		}

		if (idxF + 10 + 1 > FT.rows()) {
		    int idf = FT.rows();
		    int idl = LT.rows();
                    FT.conservativeResize(FT.rows()+Fexpandsize, FT.cols());
		    FT.block(idf, 0, Fexpandsize, FT.cols()).setConstant(0);
		    LT.conservativeResize(LT.rows()+Fexpandsize, LT.cols());
		    LT.block(idl, 0, Fexpandsize, LT.cols()).setConstant(0);
		}

		propogate_split_func(VT, idxV, FT, idxF, inD, DT, LT, uE2V, i, mpair, iso, material_threshold);

	    }

	}
    }
    //end = clock();
    //std::cout << "Propagate split func takes: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << std::endl;

    //std::cout << "VT:" << std::endl << VT << std::endl;
    //std::cout << "FT:" << std::endl << FT << std::endl;
    //std::cout << "DT:" << std::endl << DT << std::endl;

    Eigen::MatrixXi CFT;
    carve_out_material(FT, idxF, LT, DT, mid, material_threshold, CFT);
    
    //std::cout << "CFT:" << std::endl << CFT << std::endl;
    //std::cout << "VT:" << std::endl << VT << std::endl;
    //std::cout << "FT:" << std::endl << FT << std::endl;
    //std::cout << "DT:" << std::endl << DT << std::endl;

    //start = clock();
    if (idxCFT + CFT.rows() > CFTarray.rows() && idxCFT + CFT.rows() > CFTarray.cols()) {
        int r_CFTarray = CFTarray.rows();
        CFTarray.conservativeResize(2*r_CFTarray, CFTarray.cols());
	CFTarray.block(r_CFTarray, 0, r_CFTarray, CFTarray.cols()).setConstant(0);
        int r_CLTarray = CLTarray.rows();
	CLTarray.conservativeResize(2*r_CLTarray, CLTarray.cols());
	CLTarray.block(r_CLTarray, 0, r_CLTarray, CLTarray.cols()).setConstant(false);
    }

    CFTarray.block(idxCFT, 0, CFT.rows(), CFT.cols()) = CFT;
    ArrayXb CLT(CFT.rows(), DT.cols());
    CLT.setConstant(false);
    CLT.col(mid).setConstant(true);
    CLTarray.block(idxCFT, 0, CLT.rows(), CLT.cols()) = CLT;
    idxCFT += CFT.rows();
    
    //end = clock();
    //std::cout << "Finally remove intersected tets takes: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << std::endl;

  }

  VT.conservativeResize(idxV, VT.cols());
  FT = CFTarray.block(0,0,idxCFT-1,FT.cols());
  LT = CLTarray.block(0,0,idxCFT-1,LT.cols());

//  std::cout << "VT" << std::endl << VT << std::endl;
//  std::cout << "FT" << std::endl << FT << std::endl;
//  std::cout << "LT" << std::endl << LT << std::endl;

    UT = VT;
    GT = FT;
}